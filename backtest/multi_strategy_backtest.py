import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from pathlib import Path
from dataclasses import dataclass
from typing import List
import datetime

out_dir = Path('backtest_images')
out_dir.mkdir(exist_ok=True)
GOLD,GRN,RED,CYAN,WHT = '#D4A843','#00E676','#FF1744','#18FFFF','#AAAAAA'
PURP,ORNG,BLU = '#CE93D8','#FF9800','#42A5F5'

INTERVAL = '5m'; DAYS = 59

def fetch(t):
    tk = yf.Ticker(t)
    df = tk.history(interval=INTERVAL, period=f'{DAYS}d')
    df.columns=[c.lower() for c in df.columns]
    df=df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
    df.index=pd.to_datetime(df.index)
    if df.index.tz is not None: df.index=df.index.tz_convert('US/Eastern').tz_localize(None)
    return df.between_time('09:30','16:00')

print('Fetching NQ...')
df=fetch('NQ=F'); df_spy=fetch('SPY')
s=max(df.index[0],df_spy.index[0]); e=min(df.index[-1],df_spy.index[-1])
df=df.loc[s:e]; df_spy=df_spy.loc[s:e]
bh_close=df['Close'].copy(); N=len(df)

def atr(hi,lo,cl,p=14):
    tr=pd.concat([hi-lo,(hi-cl.shift()).abs(),(lo-cl.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/p,adjust=False,min_periods=p).mean()

# ============ SHARED INDICATORS ============
# VWAP
df['_d']=pd.Series(df.index.date,index=df.index)
df['_pv']=df['Close']*df['Volume']
df['_cpv']=df.groupby('_d')['_pv'].cumsum()
df['_cv']=df.groupby('_d')['Volume'].cumsum()
df['vwap']=df['_cpv']/df['_cv'].replace(0,np.nan)
df['_dev']=df['Close']-df['vwap']
df['_std']=df.groupby('_d')['_dev'].transform(lambda x:x.expanding().std().fillna(0))
for i in range(1,4):
    df[f'vwap_u{i}']=df['vwap']+i*df['_std']
    df[f'vwap_l{i}']=df['vwap']-i*df['_std']
df['vwap_d']=100*(df['Close']-df['vwap'])/df['vwap'].replace(0,np.nan)
df.drop(columns=['_d','_pv','_cpv','_cv','_dev','_std'],inplace=True)

# Greeks
c=df['Close'].astype(float)
df['delta_p']=((c-c.shift(10))/c.shift(10)).fillna(0)
df['delta_s']=df['delta_p'].ewm(span=10,adjust=False).mean()
dm=df['delta_s'].abs().rolling(20,min_periods=1).max()
df['delta_n']=(df['delta_s']/dm.replace(0,0.001)).clip(-1,1).fillna(0)
df['gamma_r']=df['delta_s'].diff().fillna(0)
gs=df['gamma_r'].rolling(20,min_periods=1).std()
df['gamma_n']=(df['gamma_r']/gs.replace(0,0.0001)).clip(-3,3).fillna(0)
df['gamma_m']=df['gamma_n'].abs().rolling(10,min_periods=1).mean()

# Cumulative Delta
d_=np.select([df['Close']>df['Close'].shift(1),df['Close']<df['Close'].shift(1)],[1,-1],default=0)
df['cd']=(d_*df['Volume']).groupby(df.index.date).cumsum()

# ATR
df['atr14']=atr(df['High'],df['Low'],df['Close'],14)

# Volume avg
df['vol_avg']=df['Volume'].rolling(50,min_periods=10).mean()

# Time filter
times=pd.Series(df.index,index=df.index).apply(lambda x:x.time())
mo=datetime.time(9,30); mc=datetime.time(16,0)
oc=mo.hour*60+mo.minute+15; cc=mc.hour*60+mc.minute-15
df['time_ok']=times.apply(lambda t:oc<=t.hour*60+t.minute<=cc)

# SPY delta
spy_c=df_spy['Close'].reindex(df.index,method='ffill')
df['spy_delta']=((spy_c-spy_c.shift(10))/spy_c.shift(10)).fillna(0).ewm(span=10,adjust=False).mean()

# ============ BACKTEST ENGINE ============
@dataclass
class Trade:
    ei:int; d:str; ep:float; sp:float; tp:float; name:str=''
    xi:int=-1; xp:float=np.nan; xr:str=''; pnl:float=0.0; bh:int=0

def sim_single(df,sig_col,direction,stop_dist,target_dist,max_bars=78):
    ts=[]; cl,hi,lo=df['Close'].values,df['High'].values,df['Low'].values
    av=df['atr14'].values; n_=len(df)
    for i in range(n_):
        if not df[sig_col].iloc[i]: continue
        ep=cl[i]; atr_i=av[i]
        if direction=='long': sp=ep-atr_i*stop_dist; tp=ep+atr_i*target_dist
        else: sp=ep+atr_i*stop_dist; tp=ep-atr_i*target_dist
        if sp<=0 or tp<=0: continue
        t=Trade(ei=i,d=direction,ep=ep,sp=sp,tp=tp)
        for j in range(i+1,min(n_,i+max_bars+1)):
            hs=lo[j]<=sp if direction=='long' else hi[j]>=sp
            ht=hi[j]>=tp if direction=='long' else lo[j]<=tp
            if hs:
                t.xi=j; t.xp=sp; t.xr='stop'; t.bh=j-i
                t.pnl=(sp/ep-1)*100 if direction=='long' else (1-sp/ep)*100; break
            elif ht:
                t.xi=j; t.xp=tp; t.xr='target'; t.bh=j-i
                t.pnl=(tp/ep-1)*100 if direction=='long' else (1-tp/ep)*100; break
        else:
            ej=min(n_-1,i+max_bars); t.xi=ej; t.xp=cl[ej]; t.xr='time'; t.bh=ej-i
            t.pnl=(cl[ej]/ep-1)*100 if direction=='long' else (1-cl[ej]/ep)*100
        ts.append(t)
    return ts

def sim_both(df,sig_l,sig_s,stop_dist,target_dist,max_bars=78):
    return sim_single(df,sig_l,'long',stop_dist,target_dist,max_bars)+\
           sim_single(df,sig_s,'short',stop_dist,target_dist,max_bars)

def dedup_sigs(sig_series,cooldown=10):
    s=sig_series.copy(); last=-cooldown-1
    for i in range(len(s)):
        if s.iloc[i]:
            if i-last<cooldown: s.iloc[i]=False
            else: last=i
    return s

def equity_curve(trades,df):
    eq=pd.Series(100.0,index=df.index); at=None
    for i in range(len(df)):
        if at is None:
            for t in trades:
                if t.ei==i: at=t; break
        if at is not None and i==at.xi:
            eq.iloc[i]=eq.iloc[i-1]*(1+at.pnl/100); at=None
        elif i>0: eq.iloc[i]=eq.iloc[i-1]
    return eq.ffill()

def compute_metrics(name,trades_list,eq_series):
    if len(trades_list)==0:
        return {'Name':name,'Trades':0,'WR':0,'Avg PnL':0,'Return':0,
                'CAGR':0,'Max DD':0,'Sharpe':0,'PF':0,'Exposure':0}
    tdf=pd.DataFrame([{'pnl':t.pnl,'xr':t.xr,'bh':t.bh,'d':t.d} for t in trades_list])
    tt=len(tdf); wr=(tdf['pnl']>0).mean()*100; ap=tdf['pnl'].mean()
    nd_=len(eq_series); ff=eq_series.iloc[-1]
    cag=(ff/100)**(252/max(nd_,1))-1
    mdd_=(eq_series/eq_series.expanding().max()-1).min()*100
    r_=tdf['pnl']/100
    sh=r_.mean()/r_.std()*np.sqrt(252*78/6.5) if r_.std()>0 else 0
    ps=tdf[tdf['pnl']>0]['pnl'].sum(); ls=abs(tdf[tdf['pnl']<=0]['pnl'].sum())
    pf=ps/ls if ls>0 else float('inf')
    ex=tdf['bh'].sum()/N*100
    return {'Name':name,'Trades':tt,'WR':wr,'Avg PnL':ap,'Return':ff-100,
            'CAGR':cag*100,'Max DD':mdd_,'Sharpe':sh,'PF':pf,'Exposure':ex}

# ============ STRATEGY 1: VWAP MEAN REVERSION ============
print('Running Strategy 1: VWAP Reversion...')
df['s1_long']=dedup_sigs((df['Close']<df['vwap_l2'])&df['time_ok'])
df['s1_short']=dedup_sigs((df['Close']>df['vwap_u2'])&df['time_ok'])
t1=sim_both(df,'s1_long','s1_short',stop_dist=1.5,target_dist=2.0,max_bars=78)
for t in t1: t.name='VWAP Reversion'
print(f'  Signals: {len(t1)}')

# ============ STRATEGY 2: OPENING RANGE BREAKOUT ============
print('Running Strategy 2: ORB...')
df['_orb_date']=df.index.date
orb_high=df.groupby('_orb_date')['High'].transform(lambda x:x.iloc[:6].max() if len(x)>=6 else x.max())
orb_low=df.groupby('_orb_date')['Low'].transform(lambda x:x.iloc[:6].min() if len(x)>=6 else x.min())
orb_range=orb_high-orb_low
after_open=pd.Series(df.index,index=df.index).apply(lambda x:x.time()>datetime.time(10,0))
df['s2_long']=dedup_sigs((df['Close']>orb_high.shift(1))&df['time_ok']&after_open)
df['s2_short']=dedup_sigs((df['Close']<orb_low.shift(1))&df['time_ok']&after_open)
t2=[]
cl,hi,lo=df['Close'].values,df['High'].values,df['Low'].values
ov_r=orb_range.values; ov_h=orb_high.values; ov_l=orb_low.values; n2=len(df)
for i in range(n2):
    for sc,dd in [('s2_long','long'),('s2_short','short')]:
        if not df[sc].iloc[i]: continue
        ep=cl[i]; orb_w=ov_r[i]
        if orb_w<=0: continue
        if dd=='long': sp=ov_l[i]; tp=ep+orb_w*2
        else: sp=ov_h[i]; tp=ep-orb_w*2
        if sp<=0 or tp<=0: continue
        t=Trade(ei=i,d=dd,ep=ep,sp=sp,tp=tp,name='ORB')
        for j in range(i+1,min(n2,i+79)):
            hs=lo[j]<=sp if dd=='long' else hi[j]>=sp
            ht=hi[j]>=tp if dd=='long' else lo[j]<=tp
            if hs:
                t.xi=j; t.xp=sp; t.xr='stop'; t.bh=j-i
                t.pnl=(sp/ep-1)*100 if dd=='long' else (1-sp/ep)*100; break
            elif ht:
                t.xi=j; t.xp=tp; t.xr='target'; t.bh=j-i
                t.pnl=(tp/ep-1)*100 if dd=='long' else (1-tp/ep)*100; break
        else:
            ej=min(n2-1,i+78); t.xi=ej; t.xp=cl[ej]; t.xr='time'; t.bh=ej-i
            t.pnl=(cl[ej]/ep-1)*100 if dd=='long' else (1-cl[ej]/ep)*100
        t2.append(t)
print(f'  Signals: {len(t2)}')

# ============ STRATEGY 3: GAP FILL ============
print('Running Strategy 3: Gap Fill...')
df['_prev_close']=df['Close'].shift(1)
df['_first_bar']=df.index.to_series().apply(lambda x:x.time())==datetime.time(9,30)
df['_gap_up']=False; df['_gap_down']=False
for i in range(len(df)):
    if df['_first_bar'].iloc[i] and i>0 and pd.notna(df['_prev_close'].iloc[i-1]):
        gap_pct=(df['Open'].iloc[i]-df['_prev_close'].iloc[i-1])/df['_prev_close'].iloc[i-1]
        end_idx=min(i+40,N-1)
        if gap_pct>0.003:
            df.iloc[i:end_idx+1,df.columns.get_loc('_gap_up')]=True
        elif gap_pct<-0.003:
            df.iloc[i:end_idx+1,df.columns.get_loc('_gap_down')]=True
after_open2=pd.Series(df.index,index=df.index).apply(lambda x:x.time()>datetime.time(9,30))
df['s3_short']=dedup_sigs(df['_gap_up']&after_open2&df['time_ok'])
df['s3_long']=dedup_sigs(df['_gap_down']&after_open2&df['time_ok'])
t3=sim_both(df,'s3_long','s3_short',stop_dist=1.0,target_dist=2.0,max_bars=40)
for t in t3: t.name='Gap Fill'
print(f'  Signals: {len(t3)}')

# ============ STRATEGY 4: CUMULATIVE DELTA DIVERGENCE ============
print('Running Strategy 4: CD Divergence...')
df['_ph20']=df['High'].rolling(20,min_periods=20).max()
df['_pl20']=df['Low'].rolling(20,min_periods=20).min()
df['_cd_h20']=df['cd'].rolling(20,min_periods=20).max()
df['_cd_l20']=df['cd'].rolling(20,min_periods=20).min()
df['_bear_div']=(df['High']>=df['_ph20'].shift(1))&(df['cd']<df['_cd_h20'].shift(5))
df['_bull_div']=(df['Low']<=df['_pl20'].shift(1))&(df['cd']>df['_cd_l20'].shift(5))
df['s4_short']=dedup_sigs(df['_bear_div']&df['time_ok'],cooldown=30)
df['s4_long']=dedup_sigs(df['_bull_div']&df['time_ok'],cooldown=30)
t4=sim_both(df,'s4_long','s4_short',stop_dist=1.5,target_dist=2.5,max_bars=78)
for t in t4: t.name='CD Divergence'
print(f'  Signals: {len(t4)}')

# ============ STRATEGY 5: VWAP PULLBACK TREND ============
print('Running Strategy 5: VWAP Pullback...')
df['_trend_up']=(df['vwap_d']>1.5)&(df['delta_n']>0.2)
df['_trend_dn']=(df['vwap_d']<-1.5)&(df['delta_n']<-0.2)
df['_pb_long']=(df['_trend_up'].rolling(10,min_periods=5).max()>0)&(df['vwap_d'].abs()<0.3)&(df['Close']>df['Close'].shift(1))
df['_pb_short']=(df['_trend_dn'].rolling(10,min_periods=5).max()>0)&(df['vwap_d'].abs()<0.3)&(df['Close']<df['Close'].shift(1))
df['s5_long']=dedup_sigs(df['_pb_long']&df['time_ok'],cooldown=20)
df['s5_short']=dedup_sigs(df['_pb_short']&df['time_ok'],cooldown=20)
t5=sim_both(df,'s5_long','s5_short',stop_dist=1.5,target_dist=3.0,max_bars=78)
for t in t5: t.name='VWAP Pullback'
print(f'  Signals: {len(t5)}')

# ============ STRATEGY 6: INSTITUTIONAL V3 ============
print('Running Strategy 6: Institutional V3...')
lb=20; rb=5; mw=0.0008; bp_=0.0005
lo,hi,cl_=df['Low'].astype(float),df['High'].astype(float),df['Close'].astype(float)
rsl=lo.rolling(lb,min_periods=lb).min(); rsh=hi.rolling(lb,min_periods=lb).max()
swl=(lo<rsl.shift(1))&(lo.shift(1)>=rsl.shift(2)); swlv=(rsl.shift(1)-lo)>(rsl.shift(1)*mw)
rc_=(cl_>rsl.shift(1)).rolling(rb,min_periods=1).max().shift(-rb+1).fillna(0)>0
df['liq_l']=(swl&swlv&rc_).fillna(False)
swh=(hi>rsh.shift(1))&(hi.shift(1)<=rsh.shift(2)); swhv=(hi-rsh.shift(1))>(rsh.shift(1)*mw)
rcs=(cl_<rsh.shift(1)).rolling(rb,min_periods=1).max().shift(-rb+1).fillna(0)>0
df['liq_s']=(swh&swhv&rcs).fillna(False)
ph=hi.rolling(lb,min_periods=1).max(); pl=lo.rolling(lb,min_periods=1).min()
df['msb_b']=(cl_>ph.shift(1))&((cl_-ph.shift(1))/ph.shift(1)>bp_)
df['msb_s']=(cl_<pl.shift(1))&((pl.shift(1)-cl_)/pl.shift(1)>bp_)
hl_=df['liq_l'].rolling(lb*2,min_periods=1).max()>0; hs_=df['liq_s'].rolling(lb*2,min_periods=1).max()>0
df['msb_bv']=df['msb_b']&hl_; df['msb_sv']=df['msb_s']&hs_

def score6(df_,dd_):
    s_=pd.Series(0.0,index=df_.index); lb2=lb*2
    if dd_=='long':
        h_=df_['liq_l'].rolling(lb2,min_periods=1).max()>0; s_+=h_.astype(float)*30
        s_+=df_['msb_bv'].astype(float)*30
        s_+=np.where(df_['vwap_d'].abs()<1.5,1,0).astype(float)*20
        s_+=((df_['spy_delta']>0.05).values).astype(float)*10
        s_+=(df_['delta_n']>0.1).astype(float)*5
    else:
        h_=df_['liq_s'].rolling(lb2,min_periods=1).max()>0; s_+=h_.astype(float)*30
        s_+=df_['msb_sv'].astype(float)*30
        s_+=np.where(df_['vwap_d'].abs()<1.5,1,0).astype(float)*20
        s_+=((df_['spy_delta']<-0.05).values).astype(float)*10
        s_+=(df_['delta_n']<-0.1).astype(float)*5
    return s_

df['sc_l']=score6(df,'long'); df['sc_s']=score6(df,'short')
df['s6_long']=dedup_sigs((df['liq_l'].rolling(lb*2,min_periods=1).max()>0)&(df['sc_l']>=70)&(df['Close']>df['Close'].shift(1))&df['time_ok'])
df['s6_short']=dedup_sigs((df['liq_s'].rolling(lb*2,min_periods=1).max()>0)&(df['sc_s']>=70)&(df['Close']<df['Close'].shift(1))&df['time_ok'])
t6=sim_both(df,'s6_long','s6_short',stop_dist=2.0,target_dist=3.0,max_bars=78)
for t in t6: t.name='Institutional V3'
print(f'  Signals: {len(t6)}')

# ============ COMPUTE ALL EQUITY CURVES ============
print('\nComputing equity curves...')
beq=bh_close/bh_close.iloc[0]*100
eqs={}
strategy_data=[
    ('Buy & Hold', [], beq),
    ('VWAP Reversion', t1, equity_curve(t1,df)),
    ('ORB', t2, equity_curve(t2,df)),
    ('Gap Fill', t3, equity_curve(t3,df)),
    ('CD Divergence', t4, equity_curve(t4,df)),
    ('VWAP Pullback', t5, equity_curve(t5,df)),
    ('Institutional V3', t6, equity_curve(t6,df)),
]
for name,_,eq in strategy_data: eqs[name]=eq
all_names=[s[0] for s in strategy_data]

# Daily resample
for k in eqs: eqs[k]=eqs[k].resample('D').last().dropna()
cdates=eqs['Buy & Hold'].index
for k in eqs: eqs[k]=eqs[k].reindex(cdates).ffill().dropna()
cdates=eqs['Buy & Hold'].index.intersection(eqs['Institutional V3'].index)
for k in eqs: eqs[k]=eqs[k].loc[cdates]
nd_=len(eqs['Buy & Hold'])

# ============ METRICS ============
all_metrics=[]
# B&H
ff_bh=eqs['Buy & Hold'].iloc[-1]; cag_bh=(ff_bh/100)**(252/max(nd_,1))-1
mdd_bh=(eqs['Buy & Hold']/eqs['Buy & Hold'].expanding().max()-1).min()*100
r_bh=eqs['Buy & Hold'].pct_change().dropna()
sh_bh=r_bh.mean()/r_bh.std()*np.sqrt(252) if r_bh.std()>0 else 0
all_metrics.append({'Name':'Buy & Hold','Trades':1,'WR':(ff_bh>100)*100,'Avg PnL':ff_bh-100,
                    'Return':ff_bh-100,'CAGR':cag_bh*100,'Max DD':mdd_bh,'Sharpe':sh_bh,
                    'PF':float('inf'),'Exposure':100})
for name,trades_list,_ in strategy_data[1:]:
    all_metrics.append(compute_metrics(name,trades_list,eqs[name]))

mdf=pd.DataFrame(all_metrics)

# ============ PLOT ============
print('Generating plots...')
colors_s={'Buy & Hold':WHT,'VWAP Reversion':'#4FC3F7','ORB':ORNG,
          'Gap Fill':PURP,'CD Divergence':'#26A69A','VWAP Pullback':'#EF5350',
          'Institutional V3':GOLD}

fig=plt.figure(figsize=(28,20))
fig.patch.set_facecolor('#0d0d1a')
gs_=gridspec.GridSpec(4,3,figure=fig,hspace=0.35,wspace=0.3,height_ratios=[2.5,1.5,1.5,1.5])

# Row 0: All equity curves
ax_eq=fig.add_subplot(gs_[0,:])
for name in all_names:
    c=colors_s.get(name,WHT)
    lw=2.5 if name=='Institutional V3' else (1.5 if name=='Buy & Hold' else 1.0)
    alpha=1.0 if name in ['Institutional V3','Buy & Hold'] else 0.7
    eq=eqs[name]
    ax_eq.plot(eq.index,eq.values,color=c,lw=lw,alpha=alpha,label=f'{name} ({eq.iloc[-1]:.1f})')
ax_eq.axhline(100,color='white',ls='--',alpha=0.2)
ax_eq.set_title('All Strategy Equity Curves (base=100)',color='white',fontsize=14,fontweight='bold')
ax_eq.legend(loc='upper left',fontsize=9,ncol=4)
ax_eq.grid(True,alpha=0.12); ax_eq.set_facecolor('#1a1a2e'); ax_eq.tick_params(colors='white')

# Rows 1-2: Individual strategy detail panels
strategy_panels=[
    ('VWAP Reversion',t1,1,0),('ORB',t2,1,1),('Gap Fill',t3,1,2),
    ('CD Divergence',t4,2,0),('VWAP Pullback',t5,2,1),('Institutional V3',t6,2,2),
]
for name,trades_list,row,col in strategy_panels:
    ax=fig.add_subplot(gs_[row,col])
    eq=eqs[name]; c=colors_s.get(name,WHT)
    ax.plot(eq.index,eq.values,color=c,lw=2)
    ax.fill_between(eq.index,100,eq.values,where=eq.values>=100,color=GRN,alpha=0.1)
    ax.fill_between(eq.index,100,eq.values,where=eq.values<100,color=RED,alpha=0.1)
    ax.axhline(100,color='white',ls='--',alpha=0.3)
    m_=compute_metrics(name,trades_list,eq)
    tt_=m_['Trades']; wr_=m_['WR']; ap_=m_['Avg PnL']; pf_=m_['PF']
    spd=tt_/nd_ if nd_>0 else 0
    ax.set_title(f'{name}\n{tt_} trades | WR={wr_:.0f}% | Avg={ap_:+.2f}% | PF={pf_:.2f} | {spd:.2f}/day',
                 color='white',fontsize=10)
    ax.grid(True,alpha=0.12); ax.set_facecolor('#1a1a2e'); ax.tick_params(colors='white',labelsize=7)

# Row 3: Metric bar charts
ax_wr=fig.add_subplot(gs_[3,0])
names_short=[m['Name'] for m in all_metrics]
wr_vals=[m['WR'] for m in all_metrics]
cb=[colors_s.get(n,WHT) for n in names_short]
bars=ax_wr.barh(range(len(names_short)),wr_vals,color=cb,alpha=0.8,edgecolor='white',lw=0.3)
for i,(bar,val) in enumerate(zip(bars,wr_vals)):
    ax_wr.text(val+1,i,f'{val:.0f}%',va='center',color='white',fontsize=8)
ax_wr.set_yticks(range(len(names_short))); ax_wr.set_yticklabels(names_short,color='white',fontsize=8)
ax_wr.set_title('Win Rate %',color='white',fontsize=12); ax_wr.set_xlim(0,115)
ax_wr.grid(True,alpha=0.12,axis='x'); ax_wr.set_facecolor('#1a1a2e'); ax_wr.tick_params(colors='white')

ax_ret=fig.add_subplot(gs_[3,1])
ret_vals=[m['Return'] for m in all_metrics]
bars=ax_ret.barh(range(len(names_short)),ret_vals,color=cb,alpha=0.8,edgecolor='white',lw=0.3)
for i,(bar,val) in enumerate(zip(bars,ret_vals)):
    ax_ret.text(val+(1 if val>=0 else -3),i,f'{val:.1f}%',va='center',color='white',fontsize=8)
ax_ret.set_yticks(range(len(names_short))); ax_ret.set_yticklabels([])
ax_ret.set_title('Return %',color='white',fontsize=12); ax_ret.axvline(0,color='white',lw=0.5)
ax_ret.grid(True,alpha=0.12,axis='x'); ax_ret.set_facecolor('#1a1a2e'); ax_ret.tick_params(colors='white')

ax_dd=fig.add_subplot(gs_[3,2])
dd_vals=[abs(m['Max DD']) for m in all_metrics]
bars=ax_dd.barh(range(len(names_short)),dd_vals,color=cb,alpha=0.8,edgecolor='white',lw=0.3)
for i,(bar,val) in enumerate(zip(bars,dd_vals)):
    ax_dd.text(val+0.5,i,f'{val:.1f}%',va='center',color='white',fontsize=8)
ax_dd.set_yticks(range(len(names_short))); ax_dd.set_yticklabels([])
ax_dd.set_title('Max Drawdown %',color='white',fontsize=12)
ax_dd.grid(True,alpha=0.12,axis='x'); ax_dd.set_facecolor('#1a1a2e'); ax_dd.tick_params(colors='white')

plt.tight_layout()
fig.savefig(out_dir/'strategy_comparison_all.png',dpi=150,facecolor='#0d0d1a',edgecolor='none')
plt.close()
print('[OK] strategy_comparison_all.png')

# ============ REPORT ============
print(f'''
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MULTI-STRATEGY BACKTEST COMPARISON                       ║
║                    NQ=F | {INTERVAL} | {nd_} trading days                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

{"Strategy":<22s} {"Trades":>6s} {"WR":>6s} {"Avg PnL":>8s} {"Return":>8s} {"Max DD":>8s} {"PF":>6s} {"Expo":>6s} {"Sigs/day":>8s}
{"─"*22} {"─"*6} {"─"*6} {"─"*8} {"─"*8} {"─"*8} {"─"*6} {"─"*6} {"─"*8}''')

for m in all_metrics:
    nm=m['Name']; tt_=m['Trades']; wr_=m['WR']; ap_=m['Avg PnL']
    ret_=m['Return']; dd_=m['Max DD']; pf_=m['PF']; ex_=m['Exposure']
    spd=tt_/nd_ if nd_>0 else 0
    pf_str=f'{pf_:.2f}' if pf_!=float('inf') else 'Inf'
    print(f'{nm:<22s} {tt_:>6d} {wr_:>5.1f}% {ap_:>+7.2f}% {ret_:>+7.2f}% {dd_:>7.2f}% {pf_str:>6s} {ex_:>5.1f}% {spd:>7.2f}')

# Best by metric
scores=[m for m in all_metrics[1:]]
best_wr=max(scores,key=lambda x:x['WR'])
best_ret=max(scores,key=lambda x:x['Return'])
best_dd=min(scores,key=lambda x:abs(x['Max DD']))
best_pf=max([m for m in scores if m['PF']!=float('inf')],key=lambda x:x['PF'])
most_sel=min(scores,key=lambda x:x['Trades'])

print(f'''
  BEST BY METRIC:
    Win Rate:        {best_wr['Name']} ({best_wr['WR']:.0f}%)
    Return:          {best_ret['Name']} ({best_ret['Return']:+.2f}%)
    Max DD:          {best_dd['Name']} ({best_dd['Max DD']:.2f}%)
    Profit Factor:   {best_pf['Name']} ({best_pf['PF']:.2f})
    Most Selective:  {most_sel['Name']} ({most_sel['Trades']} trades)

  STRATEGY DESCRIPTIONS:
    1. VWAP Reversion   — Fade extremes at VWAP bands (mean reversion)
    2. ORB              — Opening Range Breakout (30min range)
    3. Gap Fill         — Fill overnight gaps >0.3%
    4. CD Divergence    — Fade price/CD divergences (smart money fade)
    5. VWAP Pullback    — Trend continuation from VWAP pullback
    6. Institutional V3 — Liquidity + MSB + VWAP + SPY + Greeks + Time filter

  FILES:
    backtest_images/strategy_comparison_all.png

╔══════════════════════════════════════════════════════════════════════════════╗
║  Compare and combine the best elements from each strategy.                  ║
╚══════════════════════════════════════════════════════════════════════════════╝''')

mdf.to_csv(out_dir/'strategy_comparison_metrics.csv',index=False)
print('Done.')
