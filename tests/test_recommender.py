# Behaviours to protect
# 1. Unknown favourite game returns empty recommendations
# 2. Invalid favourite game ID returns empty recommendations
# 3. ✅ Recommender excludes source game itself
# 4. ✅ Games below similarity threshold are excluded
# 5. ✅ Recommender excludes all already-selected games
# 6. ✅ Recommendations are sorted by score
# 7. ✅ BecauseYouLiked explanation is populated correctly
# 8. Keep only top_k_similar neighbours

# =====================================================================


import pandas as pd
import pytest
from recommender import recommend_from_favourite_games


@pytest.fixture
def game_mappings():
    name_to_id = {
        "Catan": 1,
        "Ticket to Ride": 2,
        "Pandemic": 3,
        "Azul": 4,
    }

    id_to_name = {
        1: "Catan",
        2: "Ticket to Ride",
        3: "Pandemic",
        4: "Azul",
    }

    return name_to_id, id_to_name

@pytest.fixture
def item_neighbours():

    item_neighbours_df = pd.DataFrame(
        {
            "BGGId":[1, 1, 1],
            "SimilarBGGId":[2, 3, 4],
            "Score":[0.7, 0.9, 0.8]
            }
        )

    return item_neighbours_df


# ===========================================================================

def test_recommender_single_game_excludes_selected_game(game_mappings):
    # Arrange
    item_neighbours_df = pd.DataFrame(
        {
            "BGGId":[1, 1, 1],
            "SimilarBGGId":[1, 2, 3],
            "Score":[1.0, 0.9, 0.7]
            }
        )

    name_to_id, id_to_name = game_mappings

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


def test_recommender_multiple_game_excludes_all_selected_games(game_mappings):
    # Arrange
    item_neighbours_df = pd.DataFrame(
        {
            "BGGId":[1, 1, 1, 3, 3, 3],
            "SimilarBGGId":[1, 2, 3, 1, 3, 4],
            "Score":[1.0, 0.9, 0.7, 0.7, 1.0, 0.6]
            }
        )

    name_to_id, id_to_name = game_mappings

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


def test_recommender_exclude_games_below_similarity_threshold(game_mappings):
    # Arrange
    item_neighbours_df = pd.DataFrame(
        {
            "BGGId":[1, 1, 1],
            "SimilarBGGId":[1, 2, 3],
            "Score":[1.0, 0.9, 0.2]
            }
        )

    name_to_id, id_to_name = game_mappings

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


def test_recommendations_sorted_by_score(game_mappings):
    #Arrange
    item_neighbours_df = pd.DataFrame(
        {
            "BGGId":[1, 1, 1],
            "SimilarBGGId":[2, 3, 4],
            "Score":[0.7, 0.9, 0.8]
            }
        )

    name_to_id, id_to_name = game_mappings

    selected_games = ["Catan", "Pandemic", "Pandemic"]
    favourite_game_names = selected_games

    #Act
    recommendations = recommend_from_favourite_games(
        favourite_game_names,
        item_neighbours_df,
        name_to_id,
        id_to_name,
        similarity_threshold=0.25,
        top_n=20,
        top_k_similar=50,
        )

    #Assert
    recommended_list = list(recommendations["Name"].values)
    assert recommended_list == ["Pandemic", "Azul", "Ticket to Ride"]


def test_because_you_liked_shows_selected_games(game_mappings):
    #Arrange
    item_neighbours_df = pd.DataFrame(
        {
            "BGGId":[1, 2],
            "SimilarBGGId":[4, 4],
            "Score":[0.7, 0.4]
            }
    )

    name_to_id, id_to_name = game_mappings

    favourite_game_names = ["Catan", "Ticket to Ride"]

    #Act
    recommendations = recommend_from_favourite_games(
        favourite_game_names,
        item_neighbours_df,
        name_to_id,
        id_to_name,
        similarity_threshold=0.25,
        top_n=20,
        top_k_similar=50,
        )

    #Assert
    azul_row = recommendations[recommendations["Name"] == "Azul"].iloc[0]

    assert azul_row["BecauseYouLiked"] == "Catan, Ticket to Ride"


def test_empty_dataframe_if_no_valid_games_found(game_mappings, item_neighbours):

    # Arrange
    invalid_game = ["Bananagrams"] # expects a list
    name_to_id, id_to_name = game_mappings
    item_neighbours_df = item_neighbours

    # Act
    recommendations = recommend_from_favourite_games(
        invalid_game,
        item_neighbours_df,
        name_to_id,
        id_to_name,
        similarity_threshold=0.25,
        top_n=20,
        top_k_similar=50,
        )

    # Assert
    assert recommendations.empty


def test_empty_dataframe_if_no_recommendations(game_mappings):
    # Arrange
    favourite_game = ["Catan"]
    name_to_id, id_to_name = game_mappings

    no_result_item_neighbours_df = pd.DataFrame(
        {
            "BGGId":[1],
            "SimilarBGGId":[1],
            "Score":[1.0]
            }
    )

    # Act
    recommendations = recommend_from_favourite_games(
        favourite_game,
        no_result_item_neighbours_df,
        name_to_id,
        id_to_name,
        similarity_threshold=0.25,
        top_n=20,
        top_k_similar=50,
        )

    # Assert
    assert recommendations.empty


# ===========================================
# MOCKING

from unittest.mock import patch
from recommender import load_data

def test_load_data_creates_game_mappings():

    # Arrange
    fake_games_df = pd.DataFrame(
        {
        "BGGId": [1, 2],
        "Name": ["Catan", "Azul"],
        }
    )

    fake_neighbours_df = pd.DataFrame(
        {
        "BGGId": [1],
        "SimilarBGGId": [2],
        "Score": [0.9],
        }
    )

    # Act
    with patch(
        "recommender.pd.read_parquet",
        return_value=fake_neighbours_df
    ), patch(
        "recommender.pd.read_csv",
        return_value=fake_games_df
    ):

        item_neighbours_df, games_df, name_to_id, id_to_name = load_data()

    # Assert
    assert name_to_id == {
        "Catan": 1,
        "Azul": 2
    }

    assert id_to_name == {
        1: "Catan",
        2: "Azul"
    }
