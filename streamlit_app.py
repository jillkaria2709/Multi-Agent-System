import streamlit as st
import pandas as pd
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from crewai import Agent, Task, Crew, Process
from textwrap import dedent
import os
from langchain_openai import ChatOpenAI
from io import StringIO

# Sidebar for API key input
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Check if the API key is provided
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0.1,
        max_tokens=8000
    )
else:
    st.sidebar.error("Please enter your OpenAI API Key.")

# Define Agents (only proceed if API key is set)
if api_key:
    student_profiler = Agent(
        role='student_profiler',
        goal='From limited data, you logically deduct conclusions about students.',
        backstory='You are an expert psychologist with decades of experience.',
        llm=llm, allow_delegation=False, verbose=True
    )

    course_specialist = Agent(
        role='course specialist',
        goal='Match the suitable course to the students',
        backstory='You have exceptional knowledge of courses.',
        llm=llm, allow_delegation=False, verbose=True
    )

    Chief_Recommendation_Director = Agent(
        role="Chief Recommendation Director",
        goal=dedent("Oversee and align work with campaign goals"),
        backstory="Chief Promotion Officer of a large EdTech company.",
        llm=llm, allow_delegation=False, verbose=True
    )

    # Define Tasks
    def get_ad_campaign_task(agent, customer_description, courses):
        return Task(
            description=dedent(f"""
                You're creating a targeted marketing campaign based on the student's profile.
                This is all we know from the student customer: {customer_description}.
                These are the courses available: {courses}.
                Your task: select exactly 3 courses best suited for this customer.
            """),
            agent=agent, expected_output='A finalized marketing campaign'
        )

    # UI for Streamlit
    st.title("Personalized Course Recommendation Campaign")

    st.header("Student Profiles and Courses")

    # Input area for student profile data
    csv_input = st.text_area("Enter Student Data (CSV format)", value='''Academic Goals, Major, Hobbies, Computer Skills, Interest in Languages, GPA
    To become a software engineer, Computer Science, Gaming, Advanced, Spanish, 3.7
    To study environmental science, Environmental Science, Hiking, Intermediate, French, 3.5
    ''')

    courses_input = st.text_area("Available Courses List", value='''"Introduction to Computer Science" - Harvard University on edX
    "Biology: Life on Earth" - Coursera
    "Introduction to Psychology" - Yale University on Coursera
    "Environmental Science" - University of Leeds on FutureLearn
    ''')

    # Initialize session state for df_output
    if 'df_output' not in st.session_state:
        st.session_state.df_output = pd.DataFrame()

    # Parse CSV input to DataFrame
    if st.button("Generate Recommendations"):
        csvStringIO = StringIO(csv_input)
        df_customers = pd.read_csv(csvStringIO, sep=",")

        # Store results
        df_output_list = []

        # Process each student
        for index, row in df_customers.iterrows():
            customer_description = f"""
            Academic goals: {row['Academic Goals']}
            Major: {row[' Major']}
            Hobbies: {row[' Hobbies']}
            Computer skills: {row[' Computer Skills']}
            Language interest: {row[' Interest in Languages']}
            GPA: {row[' GPA']}
            """

            # Generate targeted courses
            task1 = get_ad_campaign_task(Chief_Recommendation_Director, customer_description, courses_input)
            
            # Crew processing
            targeting_crew = Crew(
                agents=[student_profiler, course_specialist, Chief_Recommendation_Director],
                tasks=[task1],
                process=Process.sequential
            )
            targeting_result = targeting_crew.kickoff()

            # Extract only the course names from the result
            # Extract the relevant output as a string from the CrewOutput object
            targeted_courses = str(targeting_result.output).strip() if hasattr(targeting_result, 'output') else str(targeting_result).strip()
 
            # Append result to output
            df_output_list.append({
                'Customer': customer_description,
                'Targeted Courses': targeted_courses  # Only course names
            })

        # Convert list to DataFrame and store in session state
        st.session_state.df_output = pd.DataFrame(df_output_list)

        # Display output DataFrame in Streamlit
        st.write("Generated Campaigns:")
        st.write(st.session_state.df_output)

    # Option to download the result as CSV if df_output has content
    if not st.session_state.df_output.empty:
        if st.button("Download Results as CSV"):
            csv = st.session_state.df_output.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV", data=csv, file_name="campaign_recommendations.csv", mime='text/csv')
    else:
        st.write("No data to download. Please generate recommendations first.")
