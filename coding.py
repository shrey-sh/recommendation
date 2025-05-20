import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds


# Load data
@st.cache_data
def load_data():
    students_df = pd.read_csv('dataset/students_data.csv')
    programs_df = pd.read_csv('dataset/programs_data.csv')
    history_df = pd.read_csv('dataset/application_history.csv')
    interaction_matrix = np.load('dataset/interaction_matrix.npy')

    # Clean up lists stored as strings in the dataframes
    students_df['preferred_locations'] = students_df['preferred_locations'].apply(eval)
    programs_df['fields_of_study'] = programs_df['fields_of_study'].apply(eval)

    return students_df, programs_df, history_df, interaction_matrix


# Build a collaborative filtering model
def build_cf_model(interaction_matrix):
    # Mean center the data
    interaction_matrix_mean = np.mean(interaction_matrix, axis=1)
    interaction_matrix_demeaned = interaction_matrix - interaction_matrix_mean.reshape(-1, 1)

    # SVD
    U, sigma, Vt = svds(interaction_matrix_demeaned, k=20)

    # Convert sigma to diagonal matrix
    sigma = np.diag(sigma)

    # Predict ratings
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + interaction_matrix_mean.reshape(-1, 1)

    return all_user_predicted_ratings


# Build a content-based model
def build_content_model(programs_df):
    # Extract relevant features for programs
    features = programs_df[['ranking_global', 'acceptance_rate', 'research_opportunities',
                            'industry_connections', 'scholarship_availability',
                            'alumni_employment_rate', 'international_student_percentage',
                            'tuition_fees_usd', 'duration_months']]

    # Scale features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Calculate similarity between programs
    similarity_matrix = cosine_similarity(features_scaled)

    return similarity_matrix


# Hybrid recommendation function
def get_recommendations(user_profile, programs_df, cf_predictions, content_similarity, top_n=10):
    # Content-based filtering part
    # Extract user preferences
    budget = user_profile['budget']
    preferred_locations = user_profile['preferred_locations']
    desired_specialization = user_profile['specialization']
    desired_duration = user_profile['duration']

    # Filter programs by location
    filtered_programs = programs_df[programs_df['country'].isin(preferred_locations)]

    # Filter by specialization (checking if the desired specialization is in the fields_of_study list)
    filtered_programs = filtered_programs[filtered_programs['fields_of_study'].apply(
        lambda x: desired_specialization in x)]

    # Filter by budget
    if budget == 'Low':
        filtered_programs = filtered_programs[filtered_programs['tuition_fees_usd'] < 30000]
    elif budget == 'Medium':
        filtered_programs = filtered_programs[(filtered_programs['tuition_fees_usd'] >= 30000) &
                                              (filtered_programs['tuition_fees_usd'] < 50000)]
    else:  # High
        filtered_programs = filtered_programs[filtered_programs['tuition_fees_usd'] >= 50000]

    # Filter by duration if specified
    if desired_duration != 'Any':
        filtered_programs = filtered_programs[filtered_programs['duration_months'] == int(desired_duration)]

    if len(filtered_programs) == 0:
        return pd.DataFrame(), "No programs match your criteria. Try adjusting your preferences."

    # Calculate program scores based on a weighted combination of factors
    filtered_programs['score'] = (
        # Higher ranking (lower is better) gets lower score
            (1 - (filtered_programs['ranking_global'] / 500)) * 0.15 +
            # Higher acceptance rate gets higher score
            filtered_programs['acceptance_rate'] * 0.1 +
            # Higher research opportunities get higher score
            (filtered_programs['research_opportunities'] / 10) * 0.15 +
            # Higher industry connections get higher score
            (filtered_programs['industry_connections'] / 10) * 0.15 +
            # Higher scholarship availability gets higher score
            (filtered_programs['scholarship_availability'] / 10) * 0.15 +
            # Higher employment rate gets higher score
            filtered_programs['alumni_employment_rate'] * 0.15 +
            # Higher international student percentage gets higher score
            filtered_programs['international_student_percentage'] * 0.1
    )

    # Sort by score and take top N
    top_programs = filtered_programs.sort_values('score', ascending=False).head(top_n)

    if len(top_programs) < top_n:
        message = f"Found {len(top_programs)} programs matching your criteria."
    else:
        message = f"Top {top_n} recommended programs based on your preferences:"

    # Format output
    result_df = top_programs[['program_id', 'university_name', 'program_name', 'country',
                              'duration_months', 'tuition_fees_usd', 'ranking_global',
                              'acceptance_rate', 'score']]

    return result_df, message


def main():
    st.set_page_config(page_title="Master's Program Recommender", layout="wide")

    st.title("Master's Program Recommendation System")
    st.write("""
    Find the best master's programs abroad based on your preferences and profile.
    Fill in your details below to get personalized recommendations.
    """)

    # Load data
    try:
        students_df, programs_df, history_df, interaction_matrix = load_data()

        # Build models
        cf_predictions = build_cf_model(interaction_matrix)
        content_similarity = build_content_model(programs_df)

        # Create sidebar for user inputs
        st.sidebar.header("Your Profile")

        # Get user preferences
        age = st.sidebar.slider("Age", 21, 45, 25)
        gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Non-binary"])

        # Academic background
        st.sidebar.subheader("Academic Background")
        undergrad_major = st.sidebar.selectbox("Undergraduate Major",
                                               sorted(students_df['undergraduate_major'].unique()))
        gpa = st.sidebar.slider("Undergraduate GPA (4.0 scale)", 2.5, 4.0, 3.5, 0.1)

        # Test scores
        st.sidebar.subheader("Test Scores")
        gre = st.sidebar.slider("GRE Score", 290, 340, 315)
        toefl = st.sidebar.slider("TOEFL Score", 80, 120, 100)

        # Experience
        st.sidebar.subheader("Experience")
        work_exp = st.sidebar.slider("Work Experience (years)", 0, 10, 2)
        research_exp = st.sidebar.radio("Research Experience", ["Yes", "No"])

        # Program preferences
        st.sidebar.subheader("Program Preferences")
        specialization = st.sidebar.selectbox("Desired Specialization",
                                              sorted(programs_df['fields_of_study'].explode().unique()))

        budget = st.sidebar.radio("Budget Range", ["Low", "Medium", "High"])

        duration_options = ["Any", "12", "18", "24"]
        duration = st.sidebar.selectbox("Preferred Duration (months)", duration_options)

        # Get all available countries
        all_countries = sorted(programs_df['country'].unique())
        preferred_locations = st.sidebar.multiselect("Preferred Locations", all_countries,
                                                     default=all_countries[:3])

        # Career goal
        career_goal = st.sidebar.selectbox("Career Goal",
                                           ["Industry", "Academia", "Research", "Entrepreneurship"])

        # Create user profile
        user_profile = {
            'age': age,
            'gender': gender,
            'undergraduate_major': undergrad_major,
            'undergraduate_gpa': gpa,
            'gre_score': gre,
            'toefl_score': toefl,
            'work_experience_years': work_exp,
            'research_experience': 1 if research_exp == "Yes" else 0,
            'budget': budget,
            'preferred_locations': preferred_locations,
            'specialization': specialization,
            'duration': duration,
            'career_goal': career_goal
        }

        # Add a button to get recommendations
        if st.sidebar.button("Get Recommendations"):
            # Get recommendations
            recommended_programs, message = get_recommendations(
                user_profile, programs_df, cf_predictions, content_similarity)

            st.subheader("Your Recommended Programs")
            st.write(message)

            if not recommended_programs.empty:
                # Format prices as currency
                recommended_programs['tuition_fees_usd'] = recommended_programs['tuition_fees_usd'].apply(
                    lambda x: f"${x:,.2f}")

                # Format acceptance rate as percentage
                recommended_programs['acceptance_rate'] = recommended_programs['acceptance_rate'].apply(
                    lambda x: f"{x * 100:.1f}%")

                # Rename columns for display
                display_df = recommended_programs.rename(columns={
                    'program_id': 'ID',
                    'university_name': 'University',
                    'program_name': 'Program',
                    'country': 'Country',
                    'duration_months': 'Duration (months)',
                    'tuition_fees_usd': 'Tuition',
                    'ranking_global': 'Global Ranking',
                    'acceptance_rate': 'Acceptance Rate',
                    'score': 'Match Score'
                })

                # Round score to 2 decimal places
                display_df['Match Score'] = display_df['Match Score'].apply(lambda x: f"{x:.2f}")

                # Display recommendations
                st.dataframe(display_df.drop('ID', axis=1), use_container_width=True)

                # Visualizations
                st.subheader("Program Insights")

                # Bar chart of top 5 universities by match score
                st.write("Top Universities by Match Score")
                fig, ax = plt.subplots(figsize=(10, 6))
                top5_df = recommended_programs.head(min(5, len(recommended_programs)))
                sns.barplot(x='university_name', y='score', data=top5_df, ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('University')
                plt.ylabel('Match Score')
                st.pyplot(fig)

                # Get program details for the top recommended program
                if not recommended_programs.empty:
                    top_program_id = recommended_programs.iloc[0]['program_id']
                    top_program = programs_df[programs_df['program_id'] == top_program_id].iloc[0]

                    st.subheader(f"Spotlight on: {top_program['university_name']}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Program:** {top_program['program_name']}")
                        st.write(f"**Country:** {top_program['country']}")
                        st.write(f"**Duration:** {top_program['duration_months']} months")
                        st.write(f"**Tuition:** ${top_program['tuition_fees_usd']:,.2f}")
                        st.write(f"**Global Ranking:** {top_program['ranking_global']}")

                    with col2:
                        st.write(f"**Acceptance Rate:** {top_program['acceptance_rate'] * 100:.1f}%")
                        st.write(f"**Research Opportunities:** {top_program['research_opportunities']}/10")
                        st.write(f"**Industry Connections:** {top_program['industry_connections']}/10")
                        st.write(f"**Scholarship Availability:** {top_program['scholarship_availability']}/10")
                        st.write(f"**Alumni Employment Rate:** {top_program['alumni_employment_rate'] * 100:.1f}%")
            else:
                st.write("No programs match your criteria. Try adjusting your preferences.")

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.write("Please make sure you have the required data files in the current directory.")
        st.write(
            "Required files: students_data.csv, programs_data.csv, application_history.csv, interaction_matrix.npy")


if __name__ == "__main__":
    main()