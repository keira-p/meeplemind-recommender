import pandas as pd


def load_data():
    """
    Loads precomputed data for recommender.

    Returns:
    - similarity matrix (df)
    - simple games metadata (df)
    - game ID-name mapping tools
    """
    item_similarity_df = pd.read_csv("data/processed/item_similarity.csv", index_col=0)
    games_df = pd.read_csv("data/processed/games.csv")

    # Ensure all IDs are integer type
    item_similarity_df.index = item_similarity_df.index.astype(int)
    item_similarity_df.columns = item_similarity_df.columns.astype(int)
    games_df["BGGId"] = games_df["BGGId"].astype(int)

    # Name-ID mappings
    name_to_id = dict(zip(games_df["Name"], games_df["BGGId"]))
    id_to_name = dict(zip(games_df["BGGId"], games_df["Name"]))

    return item_similarity_df, games_df, name_to_id, id_to_name


def recommend_from_favourite_games(
    favourite_game_names,
    item_similarity_df,
    name_to_id,
    id_to_name,
    similarity_threshold=0.25,
    top_n=20,
    top_k_similar=50,
):
    """
    Recommend games based on a list of favourite game names.

    Steps:
    1. Convert selected game names to BGGIds
    2. For each liked game, find similar games
    3. Keep only reasonably similar games
    4. Add up similarity scores across liked games
    5. Store which input games contributed to each recommendation
    6. Return top recommendations with simple explanations
    """

    # Convert selected names to IDs
    liked_game_ids = [
        name_to_id[name]
        for name in favourite_game_names
        if name in name_to_id
    ]

    # Keep only games that exist in similarity matrix
    liked_game_ids = [
        game_id
        for game_id in liked_game_ids
        if game_id in item_similarity_df.columns
    ]

    # Fail safely if no valid games found
    if len(liked_game_ids) == 0:
        return pd.DataFrame(columns=["BGGId", "Name", "Score", "BecauseYouLiked"])

    # Store total recommendation scores
    recommendation_scores = {}

    # Store explanation details
    recommendation_reasons = {}

    for liked_game_id in liked_game_ids:

        # Get similar games for this liked game
        similar_series = item_similarity_df[liked_game_id]

        # Remove self-match
        similar_series = similar_series.drop(liked_game_id, errors="ignore")

        # Keep strongest similar games only
        similar_series = similar_series.sort_values(ascending=False).head(top_k_similar)

        # Keep only games above similarity threshold
        similar_series = similar_series[similar_series >= similarity_threshold]

        # Loop through candidate games
        for candidate_game_id, similarity_score in similar_series.items():

            # Skip games user already selected
            if candidate_game_id in liked_game_ids:
                continue

            # Add similarity score to total
            if candidate_game_id not in recommendation_scores:
                recommendation_scores[candidate_game_id] = 0

            recommendation_scores[candidate_game_id] += similarity_score

            # Store explanation source
            if candidate_game_id not in recommendation_reasons:
                recommendation_reasons[candidate_game_id] = []

            recommendation_reasons[candidate_game_id].append(
                (liked_game_id, similarity_score)
            )

    # Convert scores to dataframe
    recommendations_df = pd.DataFrame(
        list(recommendation_scores.items()),
        columns=["BGGId", "Score"]
    )

    # Fail safely if no recommendations found
    if recommendations_df.empty:
        return pd.DataFrame(columns=["BGGId", "Name", "Score", "BecauseYouLiked"])

    # Sort and add names
    recommendations_df = recommendations_df.sort_values("Score", ascending=False)
    recommendations_df["Name"] = recommendations_df["BGGId"].map(id_to_name)

    # Build readable explanation column
    because_you_liked = []

    for candidate_game_id in recommendations_df["BGGId"]:

        reasons = recommendation_reasons[candidate_game_id]

        # Strongest contributing input games first
        reasons = sorted(reasons, key=lambda x: x[1], reverse=True)

        # Keep top 3 reasons
        top_reasons = reasons[:3]

        # Convert IDs to names
        top_reason_names = [
            id_to_name.get(liked_game_id, str(liked_game_id))
            for liked_game_id, _ in top_reasons
        ]

        because_you_liked.append(", ".join(top_reason_names))

    recommendations_df["BecauseYouLiked"] = because_you_liked

    # Drop any rows where name mapping failed
    recommendations_df = recommendations_df.dropna(subset=["Name"])

    return recommendations_df[["BGGId", "Name", "Score", "BecauseYouLiked"]].head(top_n)


# ----- TESTS -----
if __name__ == "__main__":

    # Test loading data
    print("🔄 Loading data...")

    item_similarity_df, games_df, name_to_id, id_to_name = load_data()

    print("✅ Data loaded")

    # Test recommender
    test_games = [
        games_df["Name"].iloc[0],
        games_df["Name"].iloc[1],
        games_df["Name"].iloc[2],
    ]

    print(f"\n🎯 Testing with: {test_games}")

    recommendations = recommend_from_favourite_games(
        favourite_game_names=test_games,
        item_similarity_df=item_similarity_df,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        similarity_threshold=0.25,
        top_n=10
    )

    print("\n📊 Top recommendations:")
    print(recommendations.head())
