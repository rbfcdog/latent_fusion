# Copilot Coding Instructions

These rules apply whenever the model is asked to generate code.  
They must be followed strictly.

---

# MOST IMPORTANT RULE

When generating code:

- **DO NOT write explanations**
- **DO NOT generate Markdown explaining the code**
- **DO NOT describe what the code does**
- **DO NOT summarize the solution**

Return **only the requested code**.

There must be **no text before the code** and **no text after the code**.

This rule has the **highest priority**.

---

# Output Requirements

When asked to implement something:

- Output **only code**
- Do **not include explanations**
- Do **not include Markdown explanations**
- Do **not include descriptive text**
- Do **not include reasoning**

The response must contain **nothing except the code itself**.

---

# No Comments

Code must **not contain comments**.

Do not include inline comments, block comments, or documentation comments.

---

# No Debug Output

Do **not add prints, logs, or debug output** unless the task explicitly requires program output.

---

# No Emojis

Never include emojis anywhere.

---

# No Decorative Text

Do not add any decorative or stylistic elements such as:

- banners
- separators
- titles
- section headers
- formatting meant for readability outside the code itself

---

# Minimal Output

Only include what is **strictly necessary** to implement the requested functionality.

Avoid:

- unnecessary imports
- placeholder code
- unused variables
- unnecessary scaffolding
- additional structures not required for the task

The implementation should be **direct and minimal**.