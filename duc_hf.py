from gradio_client import Client
import streamlit as st
import plotly.graph_objects as go
import json

def fetch_toxicity_levels(speech, safer):
    client = Client("duchaba/Friendly_Text_Moderation")
    result = client.predict(
            msg=speech,
            safer=safer,
            api_name="/fetch_toxicity_level"
    )
    rounded_data, removed_data = prep_data_for_visualization(json.loads(result[1]))
    return  rounded_data, removed_data


def prep_data_for_visualization(json_data):
    keys_to_remove = ["is_safer_flagged", "is_flagged", "max_key", "max_value", "sum_value", "safer_value", "message"]
    
    # Dictionary to store removed keys and their values
    removed_data = {}
    
    for key in keys_to_remove:
        if key in json_data:
            removed_data[key] = json_data.pop(key)  # Move the key-value pair to removed_data

    # Round values to 2 decimal places
    rounded_data = {key: round(value, 5) for key, value in json_data.items()}
    rounded_data_mod = {}
    for key, val in rounded_data.items():
        if "/" not in key: rounded_data_mod[key] = val

    # Print the final dictionary
    return (rounded_data_mod, removed_data)


# Streamlit app
st.title("Toxic Text Visualization")
user_input = st.sidebar.text_area("User Text")
safer = st.sidebar.text_area("Safer Level", "0.02")
# Example dictionary
# Submit button
if st.sidebar.button("Submit"):
    # Display the submitted text after clicking the button
    data, removed_data = fetch_toxicity_levels(speech=user_input, safer=float(safer))
else:
    data, removed_data = ({}, {})

if data: 
    # Create donut chart using Plotly
    labels = list(data.keys())
    values = list(data.values())

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0.4, textinfo="label", hoverinfo="label+percent")]
    )

    #fig.update_traces(marker=dict(colors=["#FFA07A", "#20B2AA", "#778899", "#FFD700"]))
    fig.update_layout(
        title="! Toxicity analysis!",
        annotations=[
            dict(
                text="",
                x=0.5,
                y=0.5,
                font_size=20,
                showarrow=False,
            )
        ],
        width=1024,  # Increase width
        height=800,  # Increase height
    )

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=False, key="chart1")


    # Display removed data as a table
    st.subheader("Extra Analysis")
    if removed_data:
        st.table(removed_data.items())
    else:
        st.write("No removed data to display.")


  



