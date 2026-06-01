# Behaviours to protect
# 1. Unknown favourite game returns empty recommendations
# 2. Invalid favourite game ID returns empty recommendations
# 3. ✅ Recommender excludes source game itself
# 4. Games below similarity threshold are excluded
# 5. ✅ Recommender excludes all already-selected games
# 6. Recommendations are sorted by score
# 7. BecauseYouLiked explanation is populated correctly

# =====================================================================


import pandas as pd
from recommender import recommend_from_favourite_games


def test_recommender_single_game_excludes_selected_game():
    # Arrange
    item_neighbours_df = pd.DataFrame(
        {
            "BGGId":[1, 1, 1],
            "SimilarBGGId":[1, 2, 3],
            "Score":[1.0, 0.9, 0.7]
            }
        )

    name_to_id = {
        "Catan": 1,
        "Ticket to Ride": 2,
        "Pandemic": 3,
        }

    id_to_name = {
        1: "Catan",
        2: "Ticket to Ride",
        3: "Pandemic",
        }

    selected_games = "Catan"
    favourite_game_names = [selected_games]

    # Act
    recommendations = recommend_from_favourite_games(
        favourite_game_names,
        item_neighbours_df,
        name_to_id,
        id_to_name,
        similarity_threshold=0.25,
        top_n=20,
        top_k_similar=50,
        )

    # Assert
    recommended_names = recommendations["Name"].values
    assert selected_games not in recommended_names


def test_recommender_multiple_game_excludes_all_selected_games():
    # Arrange
    item_neighbours_df = pd.DataFrame(
        {
            "BGGId":[1, 1, 1, 3, 3, 3],
            "SimilarBGGId":[1, 2, 3, 1, 3, 4],
            "Score":[1.0, 0.9, 0.7, 0.7, 1.0, 0.6]
            }
        )

    name_to_id = {
        "Catan": 1,
        "Ticket to Ride": 2,
        "Pandemic": 3,
        "Azul": 4
        }

    id_to_name = {
        1: "Catan",
        2: "Ticket to Ride",
        3: "Pandemic",
        4: "Azul"
        }

    selected_games = ["Catan", "Pandemic"]
    favourite_game_names = selected_games

    # Act
    recommendations = recommend_from_favourite_games(
        favourite_game_names,
        item_neighbours_df,
        name_to_id,
        id_to_name,
        similarity_threshold=0.25,
        top_n=20,
        top_k_similar=50,
        )

    # Assert
    recommended_names = recommendations["Name"].values
    for game in selected_games:
        assert game not in recommended_names


def test_recommender_exclude_games_below_similarity_threshold():
    # Arrange
    item_neighbours_df = pd.DataFrame(
        {
            "BGGId":[1, 1, 1],
            "SimilarBGGId":[1, 2, 3],
            "Score":[1.0, 0.9, 0.2]
            }
        )

    name_to_id = {
        "Catan": 1,
        "Ticket to Ride": 2,
        "Pandemic": 3,
        }

    id_to_name = {
        1: "Catan",
        2: "Ticket to Ride",
        3: "Pandemic",
        }

    # Act
    recommendations = recommend_from_favourite_games(
        favourite_game_names=["Catan"],
        item_neighbours_df=item_neighbours_df,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        similarity_threshold=0.25,
        top_n=20,
        top_k_similar=50,
        )

    # Assert
    recommended_names = recommendations["Name"].values
    assert "Ticket to Ride" in recommended_names
    assert "Pandemic" not in recommended_names
