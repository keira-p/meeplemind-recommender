import streamlit as st
from recommender import load_data, recommend_from_favourite_games

# ----- Page setup -----
st.set_page_config(page_title="MeepleMind", page_icon="🎲", layout="centered")

# ----- Load data once -----
@st.cache_data(show_spinner="Loading game data...")
def get_data():
    return load_data()

item_similarity_df, games_df, name_to_id, id_to_name = get_data()

# ----- Title + intro -----
st.title("🎲 MeepleMind")
st.write("Pick a few board games you like, and get recommendations.")

# ----- Game selection -----
game_options = sorted(games_df["Name"].dropna().unique())

selected_games = st.multiselect(
    "Choose 3–5 favourite games",
    options=game_options,
    max_selections=5
)

# ----- Recommend button -----
if st.button("Get recommendations"):
    if len(selected_games) < 1:
        st.warning("Please choose at least one game.")
    else:
        recommendations = recommend_from_favourite_games(
            favourite_game_names=selected_games,
            item_similarity_df=item_similarity_df,
            name_to_id=name_to_id,
            id_to_name=id_to_name,
            similarity_threshold=0.25,
            top_n=20,
            top_k_similar=50
        )

        if recommendations.empty:
            st.info("No recommendations found. Try a different mix of games.")
        else:
            st.subheader("Recommended for you")

            # tidy display
            display_df = recommendations.copy()
            display_df["Score"] = display_df["Score"].round(3)

            st.dataframe(
                display_df[["Name", "Score", "BecauseYouLiked"]],
                use_container_width=True,
                hide_index=True
            )
