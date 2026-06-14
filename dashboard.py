import streamlit as st
import pandas as pd
import os

st.title("🤖 AI Surveillance Dashboard")

if os.path.exists("detection_log.csv"):
    df = pd.read_csv("detection_log.csv")

    st.subheader("Detection History")
    st.dataframe(df)

    st.subheader("Detection Counts")
    st.bar_chart(df["Object"].value_counts())

    counts = df["Object"].value_counts()

    st.subheader("Object Statistics")
    st.dataframe(
        counts.reset_index().rename(
            columns={"index": "Object", "Object": "Count"}
        )
    )

else:
    st.warning("No detection data found.")

st.header("Detection Images")

image_folder = "detections"

if os.path.exists(image_folder):
    images = os.listdir(image_folder)

    if images:
        for img in sorted(images, reverse=True)[:10]:
            st.image(
                os.path.join(image_folder, img),
                caption=img,
                width=300
            )
    else:
        st.write("No detection images found.")