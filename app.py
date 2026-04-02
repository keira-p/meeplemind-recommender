import time
import streamlit as st
from recommender import load_data, recommend_from_favourite_games

# ----- Page setup -----
st.set_page_config(page_title="MeepleMind", page_icon="🎲", layout="centered")

# ----- Light CSS polish -----
st.markdown("""
<style>
/* Multiselect selected-value chips */
[data-baseweb="tag"] {
    background-color: #dce8f5 !important;
    border: 1px solid #c2d6ec !important;
    border-radius: 999px !important;
}

/* Chip text */
[data-baseweb="tag"] * {
    color: #234a74 !important;
    fill: #234a74 !important;
}

/* Selected field border */
[data-baseweb="select"] {
    border-color: #cfd8e3 !important;
}

/* Input container border */
[data-baseweb="select"] > div {
    border-color: #cfd8e3 !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# ----- Load data once -----
@st.cache_data(show_spinner="Loading game data...")
def get_data():
    return load_data()

item_similarity_df, games_df, name_to_id, id_to_name = get_data()

# ----- Popular fallback games -----
popular_games = [
    "Catan",
    "Azul",
    "Carcassonne",
    "Ticket to Ride",
    "Wingspan",
    "Pandemic",
    "7 Wonders",
    "Splendor",
    "Codenames",
    "Terraforming Mars"
]

# ----- Helper function: Scoring display -----
def score_to_label(score):
    """
    Turn raw recommendation score into a friendlier label.
    """
    if score >= 1.35:
        return "Top pick"
    elif score >= 1.15:
        return "Great match"
    else:
        return "Worth a look"

# ----- Title + intro -----
st.title("🎲 MeepleMind")
st.write("Share a few favourites and MeepleMind will suggest other games that might earn a spot on your table.")

# ----- Game selection -----
game_options = sorted(games_df["Name"].dropna().unique())

st.subheader("Choose 3–5 of your favourite games")

selected_count = len(st.session_state.get("selected_games", []))

if selected_count == 0:
    helper_text = "0/5 selected"
elif selected_count < 3:
    helper_text = f"{selected_count}/5 selected — add a few more for better recommendations"
else:
    helper_text = f"{selected_count}/5 selected — ✨ ready when you are, click outside the list to get recommendations"

st.caption(helper_text)

selected_games = st.multiselect(
    "",
    options=game_options,
    max_selections=5,
    placeholder="Start typing to find a game",
    key="selected_games"
)

st.markdown("<div style='margin-top: 10px'></div>", unsafe_allow_html=True)

# ----- Recommendations -----
is_ready = len(selected_games) >= 3

if st.button("Get recommendations", type="primary", disabled=not is_ready):
    with st.spinner("Finding recommendations..."):
        recommendations = recommend_from_favourite_games(
            favourite_game_names=selected_games,
            item_similarity_df=item_similarity_df,
            name_to_id=name_to_id,
            id_to_name=id_to_name,
            similarity_threshold=0.25,
            top_n=13,
            top_k_similar=50
        )

    # fallback: retry with looser threshold if nothing found
        if recommendations.empty:
            recommendations = recommend_from_favourite_games(
                favourite_game_names=selected_games,
                item_similarity_df=item_similarity_df,
                name_to_id=name_to_id,
                id_to_name=id_to_name,
                similarity_threshold=0.15,
                top_n=13,
                top_k_similar=50
            )
        # Tiny spinner pause
        time.sleep(0.4)

    if recommendations.empty:
        st.info(
            "You might have very unique taste — we couldn't find strong matches for this mix. Nicely done. 👀"
        )
        st.write("You could try adding one of these more widely rated favourites:")
        st.markdown("\n".join([f"- {game}" for game in popular_games]))
    else:
        st.subheader("Top picks for you")
        st.write("Here are some games you might enjoy based on your picks.")
        st.caption("Ranked by how well they match your selected games")

        # Results UI
        display_df = recommendations.copy()
        display_df["Match"] = display_df["Score"].apply(score_to_label)

        # Top 3 callouts
        top_3 = display_df.head(3).reset_index(drop=True)

        for i, row in top_3.iterrows():
            if i == 0:
                st.success(f"**#{i+1}: {row['Name']}** — {row['Match']}")
            else:
                st.info(f"**#{i+1}: {row['Name']}** — {row['Match']}")

        # Rest of results
        remaining_df = display_df.iloc[3:].copy()

        if not remaining_df.empty:
            st.markdown("### More to explore")

            st.dataframe(
                remaining_df[["Name", "Match"]],
                use_container_width=True,
                hide_index=True
            )

# ----- Info module -----
        with st.expander("ℹ️ About these results"):
            st.markdown(
                """
        **Match labels**

        - **Top pick** → especially strong match
        - **Great match** → strong recommendation
        - **Worth a look** → lighter but still relevant

        These are simplified labels to make results easier to scan — so you may see several games with the same label.

        ---

        **How it works**

        MeepleMind looks at patterns in BoardGameGeek user ratings to find games that tend to be liked by the same players.

        When you pick a few favourites, it combines those signals to suggest similar games.

        ---

        **What’s a meeple? 👀**

        A meeple is the small wooden character used in games like Carcassonne — short for “my people”.

        ---

        Data from [BoardGameGeek](https://boardgamegeek.com) — thanks to BGG for the underlying dataset.
        """
                )
