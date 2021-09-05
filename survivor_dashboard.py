# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import streamlit as st
import base64
import io




# Use the full page instead of a narrow central column
st.set_page_config(page_title='Survivor League',
                   page_icon='https://raw.githubusercontent.com/papagorgio23/Python101/master/newlogo.png',
                   layout="wide")




def get_data(sheet_id, sheet_name):
    
    sheet_name = sheet_name.replace(' ', '%20')
    
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    df = pd.read_csv(url)
    
    return df




def clean_responses(responses):
    
    df = responses.copy()
    
    # Update columns
    df.columns = ['Timestamp', 'Email', 'Double Dip', 'Name_raw', 'Week_raw', 'Pick', 'Pick 2', 'Pool_raw']
    
    
    # Clean names
    df['Name'] = df['Name_raw'].str.replace("[^a-zA-Z .'0-9]", '')
    df['Name'] = df['Name'].str.strip()
    
    # Clean weeks
    df['Week'] = df['Week_raw'].str[0:7]
    df['Week'] = np.where(df['Week']=='LOSER W', 'Week 06', df['Week'])
    
    # keep last pick only
    df = df.drop_duplicates(subset=['Name', 'Week'], keep='last')

    # set pick 2 to object
    df['Pick 2'] = df['Pick 2'].astype(object)

    df = df.drop(['Timestamp', 'Name_raw', 'Week_raw', 'Pool_raw'], 1)
    
    return df



def validate_picks(responses, double_dip):
    
    df = responses.copy()
    
    # Append double picks together
    df1 = df.loc[:,['Name', 'Week', 'Pick']]
    df2 = df.loc[:,['Name', 'Week', 'Pick 2']].rename(columns={'Pick 2' : 'Pick'})
    # df2 = df2[df2['Pick']!='N/A'] #CHECK HERE
    df = pd.concat([df1, df2])
    df = df.dropna()
    
    # Check loser picks
    loserDF1 = df[df['Week']=='Week 06']
    loserDF2 = df[df['Week'].isin(['Week 01', 'Week 02', 'Week 03', 'Week 04', 'Week 05'])]
    loserDF2 = loserDF2.rename(columns={'Week' : 'Week_'})
    losers = loserDF1.merge(loserDF2, how='left', on=['Name', 'Pick'])
    
    # Check loser eligibility
    losers['Eligible'] = np.where(losers['Week_'].isnull(), 'N', 'Y')
    
    # Count team picks (exclude losers week)
    picks = df[df['Week']!='Week 06']
    picks = picks.copy()
    picks['Count'] = picks.groupby(['Name', 'Pick']).cumcount() + 1
        
    # Check double dip
    picks = picks.merge(double_dip, how='left', on='Name')
    picks['Eligible'] = np.where(picks['Count']<2, 'Y',
                                 np.where((picks['Count']==2) & (picks['Pick']==picks['Double Dip']), 'Y', 'N'))
    
    
    # Add eligibility
    df = df.merge(losers.loc[:,['Name', 'Week', 'Eligible']], how='left', on=['Name', 'Week'])
    df = df.merge(picks.loc[:,['Name', 'Week', 'Pick', 'Eligible']], how='left', on=['Name', 'Week', 'Pick'])
    df['Eligible'] = df['Eligible_x'].fillna(df['Eligible_y'])
    df = df.drop(['Eligible_x', 'Eligible_y'], 1)
    
    
    return df



def get_results(responses, scores, picks):
    
    # Override ineligible picks
    # pick 1
    df = responses.merge(picks, how='left', on=['Name', 'Week', 'Pick'])
    df['Pick'] = np.where(df['Eligible']=='N', 'Invalid', df['Pick'])
    df = df.drop(['Eligible'], 1)
    
    
    # pick 2
    picks = picks.rename(columns={'Pick' : 'Pick 2'})
    df = df.merge(picks, how='left', on=['Name', 'Week', 'Pick 2'])
    df['Pick 2'] = np.where(df['Eligible']=='N', 'Invalid', df['Pick 2'])
    df = df.drop(['Eligible'], 1)
    
    
    # Pick scores
    df = df.merge(scores.loc[:,['Team', 'Week', 'Point_diff']],
                  how='left',
                  left_on=['Pick', 'Week'],
                  right_on=['Team', 'Week']).drop(['Team'], 1)
    
    # Double pick week scores
    df = df.merge(scores.loc[:,['Team', 'Week', 'Point_diff']],
                  how='left',
                  left_on=['Pick 2', 'Week'],
                  right_on=['Team', 'Week']).drop(['Team'], 1)
    
    # Add result
    df['Result'] = np.where(df['Point_diff_x']>0, 'W',
                            np.where(df['Point_diff_x']<0, 'L', 
                                     np.where(df['Point_diff_x']==0, 'T', 'Ongoing')))
    
    # Double pick week result 
    df['Result'] = np.where((df['Week']=='Week 14') & (df['Result']=='W') & (df['Point_diff_y']<0),
                            'L',
                            np.where((df['Pick']=='Invalid') | (df['Pick 2']=='Invalid'), 
                                      'L',
                                      df['Result']))
    
    # Total point diff
    df['Point_diff_y'] = df['Point_diff_y'].fillna(0)
    df['Point_diff'] = df['Point_diff_x'] + df['Point_diff_y']
    df = df.loc[:,['Name', 'Week', 'Pick', 'Pick 2', 'Result', 'Point_diff']]
    
    
    return df



def get_records(results, picks):
    
    picks = picks.copy()
    
    # Add eligibility
    df = results.merge(picks, how='left', on=['Name', 'Week', 'Pick'])
    picks = picks.rename(columns={'Pick' : 'Pick 2'})
    df = df.merge(picks, how='left', on=['Name', 'Week', 'Pick 2'])
    df['Eligible'] = np.where((df['Eligible_x']=='Y') & (df['Eligible_y']=='Y'), 'Y', 'N')
    df = df.drop(['Eligible_x', 'Eligible_y'], 1)
    
    # Add record
    df['W'] = np.where(df['Result']=='W', 1, 0)
    df['L'] = np.where(df['Result']=='L', 1, 0)
    df['T'] = np.where(df['Result']=='T', 1, 0)
    
    df = df.sort_values(by=['Name', 'Week'])
    df['W'] = df.groupby(['Name'])['W'].cumsum()
    df['L'] = df.groupby(['Name'])['L'].cumsum()
    df['T'] = df.groupby(['Name'])['T'].cumsum()
    
    record = df.loc[:,['Name', 'W', 'L', 'T']]
    record['Record'] = record['W'].astype(str) + '-' + record['L'].astype(str) + '-' + record['T'].astype(str)
    record = record.drop_duplicates(subset=['Name'], keep='last')
    record = record.set_index('Name')
    
    
    # Win Streaks
    streaks = df.loc[:,['Name', 'Week', 'L']]
    streaks['L1'] = np.where(streaks['L']==1, streaks['Week'], 'N/A')
    streaks['L2'] = np.where(streaks['L']==2, streaks['Week'], 'N/A')
    streaks['L3'] = np.where(streaks['L']==3, streaks['Week'], 'N/A')
    streaks['L4'] = np.where(streaks['L']==4, streaks['Week'], 'N/A')
       
    return record, streaks



def get_loss_streak(df, col):
    
    df = df[df[col]!='N/A'].groupby(['Name']).first().loc[:,[col]]
    
    return df



def get_rank(record, l1, l2, l3, l4, points, misc):
    
    
    df = record.copy()
    
    misc = misc.loc[:,['Name', 'Pool']]
    misc = misc.set_index('Name')
    misc['Pool_alt'] = np.where(misc['Pool']=='Eliminated', 'AAA', misc['Pool'])
    
    
    points = points.copy()
    points = points.groupby(['Name'])['Point_diff'].sum()
    
    df = pd.concat([record, l1, l2, l3, l4, points, misc], 1)
    df = df.sort_values(['Pool_alt', 'W', 'T', 'L', 'L1', 'L2', 'L3', 'L4', 'Point_diff'], 
                        ascending=(False, False, False, True, False, False, False, False, False))
    
    df = df.reset_index()
    df = df.reset_index().rename(columns={'index' : 'Name',
                                          'level_0' : 'Rank'})
    df['Rank'] += 1
    
    
    try:
        con_rank = df[df['Pool']=='Consolation'].iloc[0,0] - 1
    except:
        con_rank = 0
        
    try:
        elim_rank = df[df['Pool']=='Eliminated'].iloc[0,0] - 1
    except:
        elim_rank = 0
        
            
    df['Rank'] = np.where(df['Pool']=='Consolation', df['Rank'] - con_rank,
                          np.where(df['Pool']=='Eliminated', df['Rank'] - elim_rank + 300, df['Rank']))        
    
    df = df.loc[:,['Rank', 'Record', 'Name']]
    df = df.reset_index().rename(columns={'index' : 'order'})
    
    df = df.set_index('Name')
    
       
    
    return df



def generate_output(misc, rank, results):
    
    misc = misc.set_index('Name')
    results = results.copy()
    
    df = pd.concat([misc, rank], 1)
        
    
    # Generate picks pivot
    results['Pick'] = np.where(results['Pick 2'].isnull(), 
                               results['Pick'], 
                               results['Pick'] + '/' + results['Pick 2'])
    
    results = results.drop(['Pick 2', 'Point_diff'], 1)
    results['Pick'] = results['Pick'] + '_' + results['Result']
    
    results = results.pivot_table(index=['Name'],
                                  columns='Week', 
                                  values='Pick',
                                  aggfunc=lambda x: ' '.join(x))
    
    
    df = df.merge(results, how='left', left_index=True, right_index=True)
    
    # Clean up ordering
    df = df.reset_index().rename(columns={'index' : 'Name'})
    df = df.set_index(['order', 'Pool', 'Rank', 'Record', 'Name', 'Location', 'Double Dip'])
    df = df.reset_index()
    
    # Re-order
    df = df.sort_values('order')
    
    
    
    return df



def run_project():
    
    # get data
    resp = get_data('1qKpVgA7TFWp5I0VmJV3Ts6lJXaUJxpwsuP6iAHdYZOo',
                    'Form Responses 1')
    misc = get_data('1rCu_VwkkcQRbfRsK2aFo6vKieo_cD7s7fZ4J3FXXmOE',
                    'Sheet1')
    scores = get_data('13QE1fUMXkx_W3W1WBEPGN1OW3achm_fcv6Finf1JhzA',
                      'Sheet1')
    
    # clean responses
    resp = clean_responses(resp)
    
    # clean misc names
    misc['Name'] = misc['Name'].str.replace("[^a-zA-Z .'0-9]", '').str.strip()
    
    
    # validate picks
    picks = validate_picks(resp, misc.loc[:,['Name', 'Double Dip']])
    
    # get point difference
    scores['Point_diff'] = scores['Scored'] - scores['Allowed']
    
    # get pick results
    results = get_results(resp, scores, picks)
    
    # determine ranking
    record, streak = get_records(results, picks)
    streak_l1 = get_loss_streak(streak, 'L1')
    streak_l2 = get_loss_streak(streak, 'L2')
    streak_l3 = get_loss_streak(streak, 'L3')
    streak_l4 = get_loss_streak(streak, 'L4')
    rank = get_rank(record, streak_l1, streak_l2, streak_l3, streak_l4, results, misc)
    
    # generate output
    output = generate_output(misc, rank, results)
    
    
    return picks, output



picks, output = run_project()



def apply_formatting(val):
    
    if type(val) in (int, float):
        color = 'white'
    elif '_W' in val:
        color = '#00ff00'
    elif '_T' in val:
        color = '#ff9900'
    elif '_L' in val:
        color = '#ff0000'
    else:
        color = 'white'
    
    return 'background-color: %s' % color
    
        
    



st.title('Standings')
st.dataframe(output.style.applymap(apply_formatting))


# Download file
output = output.style.applymap(apply_formatting)

towrite = io.BytesIO()
downloaded_file = output.to_excel(towrite, encoding='utf-8', index=False, header=True)
towrite.seek(0)  # reset pointer
b64 = base64.b64encode(towrite.read()).decode()  # some strings
linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="survivor_standings.xlsx">Download standings</a>'
st.markdown(linko, unsafe_allow_html=True)


# Weekly view
week = st.sidebar.selectbox('Week: ', ['Week 01', 'Week 02', 'Week 03', 'Week 04', 'Week 05',
                                       'Week 06', 'Week 07', 'Week 08', 'Week 09', 'Week 10',
                                       'Week 11', 'Week 12', 'Week 13', 'Week 14', 'Week 15',
                                       'Week 16', 'Week 17', 'Week 18'])

teams = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers',
         'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos',
         'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars',
         'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins',
         'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets',
         'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers', 'Seattle Seahawks',
         'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Football Team']



st.title('Invalid Picks')
st.dataframe(picks[(picks['Week']==week) & (picks['Eligible']=='N')])

st.title('Weekly Picks')
teams = pd.DataFrame(data=teams, columns=['Pick'])
week_picks = picks[(picks['Week']==week) & (picks['Eligible']=='Y')]
team_counts = teams.merge(week_picks, how='left', on='Pick')['Pick'].value_counts()


prefix_text = "<p style='text-align: center;'>"
suffix_text = "</p>"



def transparency(team, count):
    
    count = count - 1
    
    if count > 0:
        image = """
        <table bordercolor="white" align="center"><tr><td align="center" width="9999">
        <img src="https://static.www.nfl.com/t_headshot_desktop/f_auto/league/api/clubs/logos/{0}" align="center" width="100" alt="Project icon">
        <p style='text-align: center;'>{1}</p>
        </td></tr></table>
        """
    else:
        image = """
        <table bordercolor="white" align="center"><tr><td align="center" width="9999">
        <img src="https://static.www.nfl.com/t_headshot_desktop/f_auto/league/api/clubs/logos/{0}" align="center" width="100" alt="Project icon" style="opacity:0.25">
        <p style='text-align: center;'>{1}</p>
        </td></tr></table>
        """
    
    
    return image.format(team, str(count))


row1_spacer1, row1_1, row1_2, row1_3, row1_4, row1_spacer4 = st.beta_columns((2,.75, .75, .75, .75, 2))


with row1_1:
    st.markdown(transparency('ARI', team_counts['Arizona Cardinals']), unsafe_allow_html=True)
    st.markdown(transparency('CAR', team_counts['Carolina Panthers']), unsafe_allow_html=True)
    st.markdown(transparency('DAL', team_counts['Dallas Cowboys']), unsafe_allow_html=True)
    st.markdown(transparency('HOU', team_counts['Houston Texans']), unsafe_allow_html=True)
    st.markdown(transparency('LV', team_counts['Las Vegas Raiders']), unsafe_allow_html=True)
    st.markdown(transparency('MIN', team_counts['Minnesota Vikings']), unsafe_allow_html=True)
    st.markdown(transparency('NYJ', team_counts['New York Jets']), unsafe_allow_html=True)
    st.markdown(transparency('SEA', team_counts['Seattle Seahawks']), unsafe_allow_html=True)
    
with row1_2:
    st.markdown(transparency('ATL', team_counts['Atlanta Falcons']), unsafe_allow_html=True)
    st.markdown(transparency('CHI', team_counts['Chicago Bears']), unsafe_allow_html=True)
    st.markdown(transparency('DEN', team_counts['Denver Broncos']), unsafe_allow_html=True)
    st.markdown(transparency('IND', team_counts['Indianapolis Colts']), unsafe_allow_html=True)
    st.markdown(transparency('LAC', team_counts['Los Angeles Chargers']), unsafe_allow_html=True)
    st.markdown(transparency('NE', team_counts['New England Patriots']), unsafe_allow_html=True)
    st.markdown(transparency('PHI', team_counts['Philadelphia Eagles']), unsafe_allow_html=True)
    st.markdown(transparency('TB', team_counts['Tampa Bay Buccaneers']), unsafe_allow_html=True)
with row1_3:
    st.markdown(transparency('BAL', team_counts['Baltimore Ravens']), unsafe_allow_html=True)
    st.markdown(transparency('CIN', team_counts['Cincinnati Bengals']), unsafe_allow_html=True)
    st.markdown(transparency('DET', team_counts['Detroit Lions']), unsafe_allow_html=True)
    st.markdown(transparency('JAX', team_counts['Jacksonville Jaguars']), unsafe_allow_html=True)
    st.markdown(transparency('LAR', team_counts['Los Angeles Rams']), unsafe_allow_html=True)
    st.markdown(transparency('NO', team_counts['New Orleans Saints']), unsafe_allow_html=True)
    st.markdown(transparency('PIT', team_counts['Pittsburgh Steelers']), unsafe_allow_html=True)
    st.markdown(transparency('TEN', team_counts['Tennessee Titans']), unsafe_allow_html=True)
with row1_4:
    st.markdown(transparency('BUF', team_counts['Buffalo Bills']), unsafe_allow_html=True)
    st.markdown(transparency('CLE', team_counts['Cleveland Browns']), unsafe_allow_html=True)
    st.markdown(transparency('GB', team_counts['Green Bay Packers']), unsafe_allow_html=True)
    st.markdown(transparency('KC', team_counts['Kansas City Chiefs']), unsafe_allow_html=True)
    st.markdown(transparency('MIA', team_counts['Miami Dolphins']), unsafe_allow_html=True)
    st.markdown(transparency('NYG', team_counts['New York Giants']), unsafe_allow_html=True)
    st.markdown(transparency('SF', team_counts['San Francisco 49ers']), unsafe_allow_html=True)
    st.markdown(transparency('WAS', team_counts['Washington Football Team']), unsafe_allow_html=True)




# if __name__ == '__main__':
#     test = run_project()

