import pandas as pd
import re, io, sys, glob, os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from openai import OpenAI
from dotenv import load_dotenv


df = pd.read_csv("titanic.csv")

system_template = """You are a data analysis assistant that writes Python code
to analyze the Titanic dataset with the following columns- 'PassengerId', 'Survived' with binary values 0 and 1, 'Pclass' with 1(1st class), 2(2nd class), and 3(3rd class) as possible values,
'Name', 'Sex' with male and female as values, 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', and 'Embarked' with C = Cherbourg Q = Queenstown S = Southampton.

Follow these steps:
1. Figure out what the user is asking.
2. Generate a short Python code snippet that uses the Titanic dataset to answer.
3. Write 'python' in starting of code so that we can identify it.
3. Code can include plots if relevant (using matplotlib/seaborn).
4. Return only your code snippet in triple backticks,
   and optionally a short final text explanation or numeric answer.

Below are few-shot examples (for the Titanic dataset) to guide you:

User Question: "How many males survived?"
Thought: Filter df by Sex == 'male' and Survived == 1, then count rows.
Code: result = df[(df["Sex"] == "male") & (df["Survived"] == 1)].shape[0] print("Number of males who survived:", result)
Answer: "There were 109 males who survived." # example

---

User Question: "What percentage of survivors were females?"
Thought: Among Survived == 1, find fraction that is female.
Code: survivors = df[df["Survived"] == 1] female_count = survivors[survivors["Sex"] == "female"].shape[0] percentage = (female_count / survivors.shape[0]) * 100 print(f"Percentage of survivors who were female: {percentage:.2f}%")

Answer: "About 68.13% of survivors were females." # example

---

User Question: "How many people survived?"
Thought: Survived == 1 => count.
Code:
result = df[df["Survived"] == 1].shape[0] print("Total survivors:", result)
Answer: "There were 342 survivors." # example

---

User Question: "What is the age distribution of survivors?"
Thought: We'll generate a histogram.
Code:import matplotlib.pyplot as plt import seaborn as sns
     survivors_age = df[df["Survived"] == 1]["Age"].dropna() plt.figure(figsize=(8,5)) sns.histplot(survivors_age, kde=True) plt.title("Age Distribution of Survivors") plt.xlabel("Age") plt.ylabel("Count") plt.savefig("survivors_age_dist.png") print("Plot saved: survivors_age_dist.png")
Answer: "Plot created. Most survivors are between 20 and 40 years old." # example

---
User Question: "What was the ticket fare distribution of survivors?"
Thought: Another histogram for Fare among Survived == 1.
Code:import matplotlib.pyplot as plt import seaborn as sns
     survivors_fare = df[df["Survived"] == 1]["Fare"] plt.figure(figsize=(8,5)) sns.histplot(survivors_fare, kde=True) plt.title("Fare Distribution of Survivors") plt.xlabel("Fare") plt.ylabel("Count") plt.savefig("survivors_fare_dist.png") print("Plot saved: survivors_fare_dist.png")
Answer: "Plot created. Fares among survivors cluster under $50." # example

---

User Question: "What was the ticket fare distribution of non-survivors?"
Thought: Same approach for Survived == 0.
Code: import matplotlib.pyplot as plt import seaborn as sns
      nonsurvivors_fare = df[df["Survived"] == 0]["Fare"] plt.figure(figsize=(8,5)) sns.histplot(nonsurvivors_fare, kde=True) plt.title("Fare Distribution of Non-survivors") plt.xlabel("Fare") plt.ylabel("Count") plt.savefig("nonsurvivors_fare_dist.png") print("Plot saved: nonsurvivors_fare_dist.png")
Answer: "Plot created. Most non-survivors also had fares under $50." # example

---

Now use these examples as guidance. When I ask a new question,
generate only:
1. Code snippet in triple backticks
2. A short final answer.

If it's not related to Titanic data or you can't answer, say so.
"""

def agent_respond(user_query: str) -> str:
    """
    1. Provide the system prompt (already embedded in llm_chain).
    2. Provide user_query as input in the 'human' role.
    3. Return the model's text output which should contain the code snippet + short answer.
    """
    # Because our prompt has no placeholders, we can just do:
    #   llm_chain.run(some_input)
    # But we want to treat the user query as a separate message in a chat context.
    
    # Easiest approach with LLMChain is to pass it as if it's "input" for the template. 
    # But our template has no input_variables, so let's do:
    load_dotenv()
    client=OpenAI()
    output = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_query}
        ]
    )
    return output.choices[0].message.content


def extract_code_snippet(llm_output: str) -> str:
    pattern = r"```(?:python)?(.*?)```"
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def run_user_code(code_str: str):
    """
    1. Capture stdout (print statements).
    2. Detect new .png files (plots).
    """
    old_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    before_files = set(glob.glob("*.png"))
    global_vars = {"df": df, "pd": pd, "sns": sns, "plt": plt}

    try:
        exec(code_str, global_vars)
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error executing code: {e}", []
    finally:
        sys.stdout = old_stdout

    output_text = captured_output.getvalue()
    after_files = set(glob.glob("*.png"))
    new_plots = list(after_files - before_files)

    return output_text, new_plots



def main():
    st.title("Titanic Data Analysis Assistant with Generative AI")

    if "history" not in st.session_state:
        st.session_state["history"] = []  # list of (question, code, output, plots)

    user_query = st.text_input("Ask a question about the Titanic dataset:")

    if st.button("Submit"):
        if user_query.strip():
            # 1) Call LLM to get a response with code
            llm_response = agent_respond(user_query)

            # 2) Extract the code snippet
            code_snippet = extract_code_snippet(llm_response)

            # 3) Execute code
            exec_output, plots = run_user_code(code_snippet)

            # 4) Store in session history
            st.session_state["history"].append(
                (user_query, code_snippet, exec_output, plots)
            )

    # Display entire Q&A history
    for i, (q, code, out, plot_files) in enumerate(st.session_state["history"], start=1):
        st.markdown(f"**Q{i}.** {q}")
        st.write("Generated Code:")
        st.code(code, language="python")

        st.write("**Execution Output:**")
        st.text(out)

        if plot_files:
            st.write("**Plots:**")
            for filename in plot_files:
                if os.path.exists(filename):
                    st.image(filename)
                else:
                    st.warning(f"Could not find {filename}.")
        
        st.write("---")

if __name__ == "__main__":
    main()