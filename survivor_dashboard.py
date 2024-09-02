# -*- coding: utf-8 -*-

import base64
import io

import numpy as np

# import openpyxl
import pandas as pd
import streamlit as st

# Use the full page instead of a narrow central column
st.set_page_config(
    page_title="Survivor League",
    page_icon="https://raw.githubusercontent.com/papagorgio23/Python101/master/newlogo.png",
    layout="wide",
)


class SurvivorLeague:
    def __init__(self, resp_id, resp_name, misc_id, misc_name, scores_id, scores_name):
        # self.df = df
        # self.week = week
        self.resp_id = resp_id
        self.resp_name = resp_name
        self.misc_id = misc_id
        self.misc_name = misc_name
        self.scores_id = scores_id
        self.scores_name = scores_name

    def query_data(self, sheet_id, sheet_name):
        sheet_name = sheet_name.replace(" ", "%20")

        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        df = pd.read_csv(url)

        return df

    def get_data(self):
        self.resp = self.query_data(self.resp_id, self.resp_name)

        self.misc = self.query_data(self.misc_id, self.misc_name)
        self.misc["Name"] = (
            self.misc["Name"].str.replace("[^a-zA-Z .'0-9]", "", regex=True).str.strip()
        )

        self.scores = self.query_data(self.scores_id, self.scores_name)
        self.scores["Point_diff"] = self.scores["Scored"] - self.scores["Allowed"]
        self.current_week = int(self.scores.Week.max().replace("Week ", ""))

    def clean_responses(self):
        # select appropriate columns
        self.resp = self.resp.loc[
            :,
            [
                "Timestamp",
                "Email Address",
                "Name",
                "Which Week are you Picking for? ",
                "Weekly Pick",
                "Weekly Pick #2",
            ],
        ]

        # # temp until pick 2 is added
        # self.resp["Pick 2"] = np.nan

        # Update columns
        self.resp.columns = [
            "Timestamp",
            "Email",
            "Name_raw",
            "Week_raw",
            "Pick",
            "Pick 2",
        ]

        # Clean names
        self.resp["Name"] = self.resp["Name_raw"].str.replace(
            "[^a-zA-Z .'0-9]", "", regex=True
        )
        self.resp["Name"] = self.resp["Name"].str.strip()

        # Clean weeks
        self.resp["Week"] = self.resp["Week_raw"].str[0:7]
        self.resp["Week"] = np.where(
            self.resp["Week"] == "LOSER W", "Week 06", self.resp["Week"]
        )

        # Replace nan with None string
        self.resp["Pick"] = self.resp["Pick"].fillna("None")

        # keep last pick only
        self.resp = self.resp.drop_duplicates(subset=["Name", "Week"], keep="last")

        # set pick 2 to object
        self.resp["Pick 2"] = self.resp["Pick 2"].astype(object)

        self.resp = self.resp.loc[:, ["Email", "Name", "Week", "Pick", "Pick 2"]]

    def validate_responses(self):
        double_dip = self.misc.loc[:, ["Name", "Double Dip"]]
        df = self.resp.copy(deep=True)

        # Append double picks together
        df1 = df.loc[:, ["Name", "Week", "Pick"]]
        df2 = df.loc[:, ["Name", "Week", "Pick 2"]].rename(columns={"Pick 2": "Pick"})
        df = pd.concat(objs=[df1, df2], axis=0)
        df = df.dropna()

        # Check loser picks
        loserDF1 = df[df["Week"] == "Week 06"]
        loserDF2 = df[
            df["Week"].isin(["Week 01", "Week 02", "Week 03", "Week 04", "Week 05"])
        ]
        loserDF2 = loserDF2.rename(columns={"Week": "Week_"})
        loserDF2 = loserDF2.drop_duplicates(subset=["Name", "Pick"])
        losers = loserDF1.merge(loserDF2, how="left", on=["Name", "Pick"])

        # Check loser eligibility
        losers["Eligible"] = np.where(
            (losers["Week_"].isnull()) & (losers["Pick"] != "None"), "N", "Y"
        )

        # Count team picks (exclude losers week)
        picks = df[df["Week"] != "Week 06"]
        picks = picks.copy()
        picks["Count"] = picks.groupby(["Name", "Pick"]).cumcount() + 1

        # Check double dip
        picks = picks.merge(double_dip, how="left", on="Name")
        picks["Eligible"] = np.where(
            (picks["Count"] < 2) | (picks["Pick"] == "None"),
            "Y",
            np.where(
                (picks["Count"] == 2) & (picks["Pick"] == picks["Double Dip"]), "Y", "N"
            ),
        )

        # Add eligibility
        df = df.merge(
            losers.loc[:, ["Name", "Week", "Eligible"]], how="left", on=["Name", "Week"]
        )
        df = df.merge(
            picks.loc[:, ["Name", "Week", "Pick", "Eligible"]],
            how="left",
            on=["Name", "Week", "Pick"],
        )
        df["Eligible"] = df["Eligible_x"].fillna(df["Eligible_y"])
        df = df.drop(labels=["Eligible_x", "Eligible_y"], axis=1)

        self.picks = df

    def get_results(self):
        # Override ineligible picks
        # pick 1
        df = self.resp.merge(self.picks, how="left", on=["Name", "Week", "Pick"])
        df["Pick"] = np.where(df["Eligible"] == "N", "Invalid", df["Pick"])
        df = df.drop(labels=["Eligible"], axis=1)

        # pick 2
        picks = self.picks.rename(columns={"Pick": "Pick 2"})
        df = df.merge(picks, how="left", on=["Name", "Week", "Pick 2"])
        df["Pick 2"] = np.where(df["Eligible"] == "N", "Invalid", df["Pick 2"])
        df = df.drop(labels=["Eligible"], axis=1)

        # Pick scores
        df = df.merge(
            self.scores.loc[:, ["Team", "Week", "Point_diff"]],
            how="left",
            left_on=["Pick", "Week"],
            right_on=["Team", "Week"],
        ).drop(labels=["Team"], axis=1)

        # Double pick week scores
        df = df.merge(
            self.scores.loc[:, ["Team", "Week", "Point_diff"]],
            how="left",
            left_on=["Pick 2", "Week"],
            right_on=["Team", "Week"],
        ).drop(labels=["Team"], axis=1)

        # Add result
        df["Result"] = np.where(
            df["Pick"].isin(["None", "Invalid"]),
            "L",
            np.where(
                df["Point_diff_x"] > 0,
                "W",
                np.where(
                    df["Point_diff_x"] < 0,
                    "L",
                    np.where(df["Point_diff_x"] == 0, "T", "Ongoing"),
                ),
            ),
        )

        # Double pick week result
        df["Result"] = np.where(
            (df["Week"] == "Week 15")
            & (df["Result"] == "W")  # noqa: W503
            & (df["Point_diff_y"] < 0),  # noqa: W503
            "L",
            np.where(
                (df["Pick"] == "Invalid") | (df["Pick 2"] == "Invalid"),
                "L",
                df["Result"],
            ),
        )

        # Total point diff
        df["Point_diff_y"] = df["Point_diff_y"].fillna(0)
        df["Point_diff"] = df["Point_diff_x"] + df["Point_diff_y"]
        df = df.loc[:, ["Name", "Week", "Pick", "Pick 2", "Result", "Point_diff"]]

        self.results = df

    def get_records(self):
        picks = self.picks.copy()

        # Add eligibility
        df = self.results.merge(picks, how="left", on=["Name", "Week", "Pick"])
        picks = picks.rename(columns={"Pick": "Pick 2"})
        df = df.merge(picks, how="left", on=["Name", "Week", "Pick 2"])
        df["Eligible"] = np.where(
            (df["Eligible_x"] == "Y") & (df["Eligible_y"] == "Y"), "Y", "N"
        )
        df = df.drop(labels=["Eligible_x", "Eligible_y"], axis=1)

        # Add record
        df["W"] = np.where(df["Result"] == "W", 1, 0)
        df["L"] = np.where(df["Result"] == "L", 1, 0)
        df["T"] = np.where(df["Result"] == "T", 1, 0)

        df = df.sort_values(by=["Name", "Week"])
        df["W"] = df.groupby(["Name"])["W"].cumsum()
        df["L"] = df.groupby(["Name"])["L"].cumsum()
        df["T"] = df.groupby(["Name"])["T"].cumsum()

        record = df.loc[:, ["Name", "W", "L", "T"]]
        record["Record"] = (
            record["W"].astype(str)
            + "-"  # noqa: W503
            + record["L"].astype(str)  # noqa: W503
            + "-"  # noqa: W503
            + record["T"].astype(str)  # noqa: W503
        )
        record = record.drop_duplicates(subset=["Name"], keep="last")
        record = record.set_index("Name")

        # Win Streaks
        streaks = df.loc[:, ["Name", "Week", "L"]]
        streaks["L1"] = np.where(
            streaks["L"] == 1,
            streaks["Week"].str.replace("Week ", "").astype(int),
            0,
        )
        streaks["L2"] = np.where(
            streaks["L"] == 2, streaks["Week"].str.replace("Week ", "").astype(int), 0
        )
        streaks["L3"] = np.where(
            streaks["L"] == 3, streaks["Week"].str.replace("Week ", "").astype(int), 0
        )
        streaks["L4"] = np.where(
            streaks["L"] == 4, streaks["Week"].str.replace("Week ", "").astype(int), 0
        )
        streaks["L5"] = np.where(
            streaks["L"] == 5, streaks["Week"].str.replace("Week ", "").astype(int), 0
        )
        streaks["L6"] = np.where(
            streaks["L"] == 6, streaks["Week"].str.replace("Week ", "").astype(int), 0
        )

        self.record = record
        self.streaks = streaks

    def get_loss_streak(self, col):
        df = (
            self.streaks[self.streaks[col] != 0].groupby(["Name"]).first().loc[:, [col]]
        )
        return df

    def get_rank(self):
        df = self.record.copy()

        misc = self.misc.loc[:, ["Name", "Pool"]]
        misc = misc.set_index("Name")
        misc["Pool_alt"] = np.where(misc["Pool"] == "Eliminated", "AAA", misc["Pool"])

        points = self.results.copy()
        points = points.groupby(["Name"])["Point_diff"].sum()

        df = pd.concat(
            objs=[
                self.record,
                self.streak_l1,
                self.streak_l2,
                self.streak_l3,
                self.streak_l4,
                self.streak_l5,
                self.streak_l6,
                points,
                misc,
            ],
            axis=1,
        )

        df[["L1", "L2", "L3", "L4", "L5", "L6"]] = df[
            ["L1", "L2", "L3", "L4", "L5", "L6"]
        ].fillna(self.current_week)
        df["consolation"] = df.L3 - df.L2
        df["consolation"] = np.where(
            df.L4 - df.L3 > df.consolation, df.L4 - df.L3, df.consolation
        )
        df["consolation"] = np.where(
            df.L5 - df.L4 > df.consolation, df.L5 - df.L4, df.consolation
        )
        df["consolation"] = np.where(
            df.L6 - df.L5 > df.consolation, df.L6 - df.L5, df.consolation
        )
        df["consolation"] = np.where(df.Pool == "Consolation", df.consolation, 0)

        df = df.sort_values(
            [
                "Pool_alt",
                "W",
                "T",
                "L",
                "consolation",
                "L1",
                "L2",
                "L3",
                "L4",
                "Point_diff",
            ],
            ascending=(
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ),
        )

        df = df.reset_index()
        df = df.reset_index().rename(columns={"index": "Rank"})
        df["Rank"] += 1

        try:
            con_rank = df[df["Pool"] == "Consolation"].iloc[0, 0] - 1
        except IndexError:
            con_rank = 0

        try:
            elim_rank = df[df["Pool"] == "Eliminated"].iloc[0, 0] - 1
        except IndexError:
            elim_rank = 0

        df["Rank"] = np.where(
            df["Pool"] == "Consolation",
            df["Rank"] - con_rank,
            np.where(
                df["Pool"] == "Eliminated", df["Rank"] - elim_rank + 300, df["Rank"]
            ),
        )

        df = df.loc[:, ["Rank", "Record", "Name", "Point_diff"]]
        df = df.reset_index().rename(columns={"index": "order"})

        df = df.set_index("Name")

        self.rank = df

    def generate_output(self):
        misc = self.misc.set_index("Name")
        results = self.results.copy()

        df = pd.concat(objs=[misc, self.rank], axis=1)

        # Generate picks pivot
        results["Pick"] = np.where(
            results["Pick 2"].isnull(),
            results["Pick"],
            results["Pick"] + "/" + results["Pick 2"],
        )

        results = results.drop(labels=["Pick 2", "Point_diff"], axis=1)
        results["Pick"] = results["Pick"] + "_" + results["Result"]

        results = results.pivot_table(
            index=["Name"], columns="Week", values="Pick", aggfunc=lambda x: " ".join(x)
        )

        df = df.merge(results, how="left", left_index=True, right_index=True)

        df["Point_diff"] = df["Point_diff"].fillna(0)
        df["Point_diff"] = df["Point_diff"].astype(int)
        df = df.rename(columns={"Point_diff": "Point Diff"})

        # Clean up ordering
        df = df.reset_index().rename(columns={"index": "Name"})
        df = df.set_index(
            ["order", "Pool", "Rank", "Record", "Name", "Location", "Double Dip"]
        )
        df = df.reset_index()

        # Re-order
        df = df.sort_values("order")

        self.output = df

    def run(self):
        self.get_data()
        self.clean_responses()
        self.validate_responses()
        self.get_results()
        self.get_records()
        self.streak_l1 = self.get_loss_streak("L1")
        self.streak_l2 = self.get_loss_streak("L2")
        self.streak_l3 = self.get_loss_streak("L3")
        self.streak_l4 = self.get_loss_streak("L4")
        self.streak_l5 = self.get_loss_streak("L5")
        self.streak_l6 = self.get_loss_streak("L6")
        self.get_rank()
        self.generate_output()

        return self.picks, self.output, self.misc


survivor = SurvivorLeague(
    resp_id="18FuIgOjBLXHYm1bPxIeV7qxieF3LIMlWWIgDz0GadlE",
    resp_name="Form Responses 1",
    misc_id="1HsgL1rpguUfByBjFHGRkwu7wKkk5IGXP7zmGvhjS0io",
    misc_name="Sheet1",
    scores_id="1kBlwHj7xnxGWByo5QPYcxX-50Rz-QOHSDqee-Psr75Q",
    scores_name="Sheet1",
)
picks, output, misc = survivor.run()


def apply_formatting(val):
    if type(val) in (int, float):
        color = "white"
    elif "_W" in val:
        color = "#00ff00"
    elif "_T" in val:
        color = "#ff9900"
    elif "_L" in val:
        color = "#ff0000"
    else:
        color = "white"

    return "background-color: %s" % color


st.title("Standings")
# st.dataframe(output.style.applymap(apply_formatting))
st.dataframe(output.style.map(apply_formatting))


# Download file
# output = output.style.applymap(apply_formatting)
output = output.style.map(apply_formatting)

towrite = io.BytesIO()
downloaded_file = output.to_excel(towrite, encoding="utf-8", index=False, header=True)
towrite.seek(0)  # reset pointer
b64 = base64.b64encode(towrite.read()).decode()  # some strings
linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="survivor_standings.xlsx">Download standings</a>'  # noqa: E501
st.markdown(linko, unsafe_allow_html=True)

# Weekly view
week = st.sidebar.selectbox(
    "Week: ",
    [
        "Week 01",
        "Week 02",
        "Week 03",
        "Week 04",
        "Week 05",
        "Week 06",
        "Week 07",
        "Week 08",
        "Week 09",
        "Week 10",
        "Week 11",
        "Week 12",
        "Week 13",
        "Week 14",
        "Week 15",
        "Week 16",
        "Week 17",
        "Week 18",
    ],
)

teams = [
    "Arizona Cardinals",
    "Atlanta Falcons",
    "Baltimore Ravens",
    "Buffalo Bills",
    "Carolina Panthers",
    "Chicago Bears",
    "Cincinnati Bengals",
    "Cleveland Browns",
    "Dallas Cowboys",
    "Denver Broncos",
    "Detroit Lions",
    "Green Bay Packers",
    "Houston Texans",
    "Indianapolis Colts",
    "Jacksonville Jaguars",
    "Kansas City Chiefs",
    "Las Vegas Raiders",
    "Los Angeles Chargers",
    "Los Angeles Rams",
    "Miami Dolphins",
    "Minnesota Vikings",
    "New England Patriots",
    "New Orleans Saints",
    "New York Giants",
    "New York Jets",
    "Philadelphia Eagles",
    "Pittsburgh Steelers",
    "San Francisco 49ers",
    "Seattle Seahawks",
    "Tampa Bay Buccaneers",
    "Tennessee Titans",
    "Washington Commanders",
]

st.title("Invalid Picks")
st.dataframe(picks[(picks["Week"] == week) & (picks["Eligible"] == "N")])


st.title("Missing Picks")
week_picks = picks[(picks["Week"] == week) & (picks["Eligible"] == "Y")]
missing_pickers = misc[misc["Pool"] != "Eliminated"].copy()
missing_pickers = missing_pickers.loc[:, ["Name"]].merge(
    week_picks.loc[:, ["Name", "Pick"]], how="left", on="Name"
)
missing_pickers = missing_pickers[missing_pickers["Pick"].isnull()]
st.dataframe(missing_pickers)


st.title("Weekly Picks")
teams = pd.DataFrame(data=teams, columns=["Pick"])

team_counts = week_picks["Pick"].value_counts().reset_index()
team_counts.columns = ["Pick", "Count"]
team_counts = teams.merge(team_counts, how="left", on="Pick")
team_counts = team_counts.set_index("Pick")
team_counts["Count"] = team_counts["Count"].fillna(0)
team_counts = team_counts.squeeze()

st.dataframe(team_counts)

st.title("Weekly Picks Picture")
prefix_text = "<p style='text-align: center;'>"
suffix_text = "</p>"


def transparency(team, count):
    count = int(count)

    if count > 0:
        image = """
        <table bordercolor="white" align="center"><tr><td align="center" width="9999">
        <img src="https://static.www.nfl.com/t_headshot_desktop/f_auto/league/api/clubs/logos/{0}" align="center" width="100" alt="Project icon">
        <p style='text-align: center;'>{1}</p>
        </td></tr></table>
        """  # noqa: E501
    else:
        image = """
        <table bordercolor="white" align="center"><tr><td align="center" width="9999">
        <img src="https://static.www.nfl.com/t_headshot_desktop/f_auto/league/api/clubs/logos/{0}" align="center" width="100" alt="Project icon" style="opacity:0.25">
        <p style='text-align: center;'>{1}</p>
        </td></tr></table>
        """  # noqa: E501

    return image.format(team, str(count))


row1_spacer1, row1_1, row1_2, row1_3, row1_4, row1_spacer4 = st.columns(
    (2, 0.75, 0.75, 0.75, 0.75, 2)
)


with row1_1:
    st.markdown(
        transparency("ARI", team_counts["Arizona Cardinals"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("CAR", team_counts["Carolina Panthers"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("DAL", team_counts["Dallas Cowboys"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("HOU", team_counts["Houston Texans"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("LV", team_counts["Las Vegas Raiders"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("MIN", team_counts["Minnesota Vikings"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("NYJ", team_counts["New York Jets"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("SEA", team_counts["Seattle Seahawks"]), unsafe_allow_html=True
    )

with row1_2:
    st.markdown(
        transparency("ATL", team_counts["Atlanta Falcons"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("CHI", team_counts["Chicago Bears"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("DEN", team_counts["Denver Broncos"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("IND", team_counts["Indianapolis Colts"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("LAC", team_counts["Los Angeles Chargers"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("NE", team_counts["New England Patriots"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("PHI", team_counts["Philadelphia Eagles"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("TB", team_counts["Tampa Bay Buccaneers"]), unsafe_allow_html=True
    )
with row1_3:
    st.markdown(
        transparency("BAL", team_counts["Baltimore Ravens"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("CIN", team_counts["Cincinnati Bengals"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("DET", team_counts["Detroit Lions"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("JAX", team_counts["Jacksonville Jaguars"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("LAR", team_counts["Los Angeles Rams"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("NO", team_counts["New Orleans Saints"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("PIT", team_counts["Pittsburgh Steelers"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("TEN", team_counts["Tennessee Titans"]), unsafe_allow_html=True
    )
with row1_4:
    st.markdown(
        transparency("BUF", team_counts["Buffalo Bills"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("CLE", team_counts["Cleveland Browns"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("GB", team_counts["Green Bay Packers"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("KC", team_counts["Kansas City Chiefs"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("MIA", team_counts["Miami Dolphins"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("NYG", team_counts["New York Giants"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("SF", team_counts["San Francisco 49ers"]), unsafe_allow_html=True
    )
    st.markdown(
        transparency("WAS", team_counts["Washington Commanders"]),
        unsafe_allow_html=True,
    )
