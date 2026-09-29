import pandas as pd 
import streamlit as st 

st.set_page_config(
    page_title="MindScope | Mental Health Predictor",
    page_icon="🧠",
    layout="wide"
)

st.title("How's your digital rhythm treating you?")

st.markdown(
    "Explore how social media habits, sleep, study time, "
    "and daily lifestyle relate to a predicted mental health score."
)

st.divider()


# Main layout
left_col, right_col = st.columns([1.4, 1])

# Left section: Input form
with left_col:
    with st.container(border=True):
        st.subheader("👤 About You")
        st.caption("Tell us a little about yourself.")

        st.write("Personal and academic details will appear here.")

    with st.container(border=True):
        st.subheader("🌿 Daily Habits")
        st.caption("Your sleep, study and physical activity.")

        st.write("Lifestyle inputs will appear here.")

    with st.container(border=True):
        st.subheader("📱 Digital Habits")
        st.caption("Your social media usage patterns.")

        st.write("Social media inputs will appear here.")

# Right section: Prediction result
with right_col:
    with st.container(border=True):
        st.subheader("✨ Your Prediction")

        st.write("Your predicted score will appear here.")

        st.info(
            "Fill in the form and generate a prediction "
            "to see your result."
        )

        
with left_col:
    with st.container(border=True):
        st.subheader("👤 About You")
        st.caption("Tell us a little about yourself.")

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=20,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"]
        )

        country = st.text_input(
            "Country",
            value="India",
            placeholder="Enter your country"
        )

        academic_level = st.selectbox(
            "Academic Level",
            [
                "High School",
                "Undergraduate",
                "Graduate",
                "Postgraduate"
            ]
        )

        
    with st.container(border=True):
        st.subheader("🌿 Daily Habits")
        st.caption("Tell us about your daily routine.")

        study_hours = st.number_input(
            "Study hours per day",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.5
        )

        physical_activity_hours = st.number_input(
            "Physical activity hours per day",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.5
        )

        sleep_hours_per_night = st.number_input(
            "Sleep hours per night",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

        stress_level = st.selectbox(
            "Stress level",
            ['Medium','Low','High','Very High']
        )

        
    with st.container(border=True):
        st.subheader("📱 Digital Habits")
        st.caption("Tell us about your social media usage.")

        avg_daily_usage_hours = st.number_input(
            "Average daily social media usage (hours)",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5
        )

        most_used_platform = st.selectbox(
            "Most-used platform",
            [
                "Instagram",
                "TikTok",
                "YouTube",
                "Snapchat",
                "Twitter",
                "Facebook",
                "Other"
            ]
        )

        purpose_of_use = st.selectbox(
            "Main purpose of use",
            [
                "Entertainment",
                "Education",
                "News",
                "Communication",
                "Other"
            ]
        )

        
import pandas as pd
import joblib
from pathlib import Path



MODEL_PATH = Path(__file__).parent / "social_media_rf_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()


with right_col:
    with st.container(border=True):
        st.subheader("✨ Your Prediction")
        st.caption("Your model-generated result will appear here.")

        if st.button(
            "Predict Mental Health Score",
            type="primary",
            use_container_width=True
        ):
            input_data = pd.DataFrame([{
                "age": age,
                "gender": gender,
                "country": country,
                "academic_level": academic_level,
                "most_used_platform": most_used_platform,
                "purpose_of_use": purpose_of_use,
                "avg_daily_usage_hours": avg_daily_usage_hours,
                "study_hours": study_hours,
                "physical_activity_hours": physical_activity_hours,
                "sleep_hours_per_night": sleep_hours_per_night,
                "stress_level": stress_level
            }])

            input_data = input_data[[
                "age",
                "gender",
                "country",
                "academic_level",
                "most_used_platform",
                "purpose_of_use",
                "avg_daily_usage_hours",
                "study_hours",
                "physical_activity_hours",
                "sleep_hours_per_night",
                "stress_level"
            ]]

            try:
                prediction = model.predict(input_data)[0]

                st.success("Prediction generated successfully!")

                st.metric(
                    label="Predicted Mental Health Score",
                    value=f"{prediction:.2f}"
                )

                with st.expander("View submitted inputs"):
                    st.dataframe(
                        input_data.T.rename(
                            columns={0: "Input Value"}
                        ),
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.info(
            "This is a machine-learning estimate, not a clinical "
            "diagnosis or a definitive assessment of mental health."
        )



