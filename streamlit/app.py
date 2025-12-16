import streamlit as st


# st.write('In this section you can learn many plots for visualizing data')

pages = {
    "Visualizations": [
        # st.Page("amount.py", title="Amount"),
        # st.Page("distribution.py", title="Distribution"),
        # st.Page("proportion.py", title="Proportion"),
        # st.Page("relationship.py", title="X-Y Relationship"),
        # st.Page("geospatial.py", title="Geospatial"),
    ],
}

pg = st.navigation(pages)
pg.run()