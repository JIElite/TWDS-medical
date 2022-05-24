def invalid_col_val_msg(colname, val, additional=''):
    return f'Invalid value in {colname}: {val}{additional}'


def preprocess_DIABAGE3(df):
    df['DIABAGE3_N'] = df['DIABAGE3']
    df.loc[(df['DIABAGE3'] > 97.5) | (df['DIABAGE3'].isna()), 'DIABAGE3_N'] = -1.
    
    df.loc[(df['DIABAGE3'] > 0.) & (df['DIABAGE3'] < 97.5), 'DIABAGE3_C'] = 1
    df.loc[(df['DIABAGE3'] > 97.5) & (df['DIABAGE3'] < 98.5), 'DIABAGE3_C'] = 7
    df.loc[(df['DIABAGE3'] > 98.5) & (df['DIABAGE3'] < 99.5), 'DIABAGE3_C'] = 9
    df.loc[df['DIABAGE3'].isna(), 'DIABAGE3_C'] = 0
    df['DIABAGE3_C'] = df['DIABAGE3_C'].astype('category')
    return df


def convert_FEETCHK3_numeric(value):
    if 101 <= value <= 199:
        return value * 365
    elif 201 <= value <= 299: # Here, we assume there are 52 weeks in a year
        return value * 52
    elif 301 <= value <= 399:
        return value * 12
    elif 401 <= value <= 499:
        return value
    else:
        return -1

def convert_FEETCHK3_categorical(value):
    # 200, 300, 400 are not in the original
    # dataset, so we can assume the range of
    # valid FEETCHK3 in [101, 499]
    if np.isnan(value):
        return 0
    elif 101 <= value <= 499:
        return 1
    elif value == 555:
        return 5
    elif value == 777:
        return 7
    elif value == 888:
        return 8
    elif value == 999:
        return 9
    else:
        raise ValueError(invalid_col_val_msg('FEETCHK3', value))


def convert_CHILDREN_numeric(n_children):
    # Note:
    # There is no zero children case
    # df.loc[df['CHILDREN'] == 0] : 0 rows
    if 0 <= n_children <= 87:
        return n_children
    else:
        return -1
    
def convert_CHILDREN_categorical(n_children):
    if np.isnan(n_children):
        return 0
    elif 1 <= n_children <= 87:
        return 1
    elif n_children == 88:
        return 8
    elif n_children == 99:
        return 9
    else:
        raise ValueError(invalid_col_val_msg('CHILDREN', n_children))


def preprocess_QSTLANG(df):
    """Preprocess QSTLANG and drop the infrequent value.
    
    In the original dataset, there are 387,478 questionnaire in English,
    and 14,479 in Spanish. Besides, only 1 questionnare is in neither Eng. nor Span.,
    so I decide to drop such data.
    
    The original value for:
    - English: 1
    - Spanish: 2
    Because they are already the categorical value, we don't need to do
    futher transformation
    """
    
    # There is only one value not in [1, 2] in 'QSTLANG' columns.
    # Keep the frequent values in the dataframe is equivalent to dropping
    # the infrequent one.
    #
    # df.loc[~df['QSTLANG'].isin([1, 2])]['QSTLANG']
    # row index: 206759
    df = df.loc[df['QSTLANG'].isin([1, 2])]
    df['QSTLANG'] = df['QSTLANG'].astype('category')
    return df


def convert_HHADULT_numeric(n_adults):
    if 1 <= n_adults <= 76:
        return n_adults
    else:
        return -1

def convert_HHADULT_categorical(n_adults):
    if np.isnan(n_adults): # Not asked or missing
        return 0
    elif 1 <= n_adults <= 76:
        return 1
    elif n_adults == 77: # Don't know/Not sure
        return 7
    elif n_adults == 99: # Refused
        return 9
    else:
        raise ValueError(invalid_col_val_msg('HHADULT', n_adults))


def convert_HTM4_numeric(height):
    """Remains the valid height(cm) and convert empty to -1
    
    Orinally, the definition of HTM4 in the raw data is the height in
    "meter" rather than "centimeter". However, the valid height values in
    raw data are recorded in [91, 244] which is a range value in "centimeter".
    So, I decide to keep the original valid height in the raw data.
    """
    if 91 <= height <= 244:
        return height
    else:
        return -1

def convert_HTM4_categorical(height):
    if np.isnan(height):
        return 0
    elif 91 <= height <= 244:
        return 1
    else:
        raise ValueError(invalid_col_val_msg('HTM4', height))


def convert_MARIJAN1_numeric(times):
    """Convert the frequency of using marijuana.
    
    Question:21994.0
    During the past 30 days, on how many days did you
    use marijuana or cannabis?
    
    Note:
    Although the person never used marijuana(大麻) is denoted "88",
    here, we convert the value to 0.
    """
    if 1 <= times <= 30:
        return times
    elif times == 88:
        return 0
    else:
        return -1

def convert_MARIJAN1_categorical(times):
    if np.isnan(times): # Not asked or missing
        return 0
    elif 1 <= times <= 30 or times == 88:
        return 1
    elif times == 77: # Don't know/Not sure
        return 7
    elif times == 99: # Refused
        return 9
    else:
        raise ValueError(invalid_col_val_msg('MARIJAN1', times))


def preprocess_HIVTSTD3_datetime(df):
    df['HIVTSTD3_month'] = df['HIVTSTD3'].apply(convert_HIVTSTD3_month).astype('category')
    df['HIVTSTD3_year']  = df['HIVTSTD3'].apply(convert_HIVTSTD3_year).astype('category')
    df['HIVTSTD3_C'] = df['HIVTSTD3'].apply(convert_HIVTSTD3_categorical).astype('category')
    return df

def check_and_convert_int(month_year):
    if not isinstance(month_year, int):
        month_year = int(month_year)
    return month_year

def convert_HIVTSTD3_month(month_year):
    """
    Note:
    Corner Error example: missing month (already implemented)
    - HIVTSTD3: 772019
    - HIVTSTD3: 772017
    - HIVTSTD3: 772005
    """
    if np.isnan(month_year):
        return -1
    
    month_year = check_and_convert_int(month_year)
    
    month_year = str(month_year)
    if len(month_year) == 5:
        month = int(month_year[0])
        return month if 1 <= month <= 9 else -1
    elif len(month_year) == 6:
        month = int(month_year[:1])
        return month if 10 <= month <= 12 else -1
    else:
        raise ValueError(invalid_col_val_msg('HIVTSTD3', month_year,\
                         additional=' (converting month error)'))

def convert_HIVTSTD3_year(month_year):
    if np.isnan(month_year):
        return -1
    
    month_year = check_and_convert_int(month_year)
    
    month_year = str(month_year)
    if len(month_year) == 5:
        year = int(month_year[1:])
        return year if 1985 <= year <= 2021 else -1
    elif len(month_year) == 6:
        year = int(month_year[2:])
        return year if 1985 <= year <= 2021 else -1
    else:
        raise ValueError(invalid_col_val_msg('HIVTSTD3', month_year,\
                         additional=' (converting year error)'))

def convert_HIVTSTD3_categorical(month_year):
    """
    Note:
    Corner Error example: missing month (already implemented)
    - HIVTSTD3: 772019
    - HIVTSTD3: 772017
    - HIVTSTD3: 772005
    """
    if np.isnan(month_year):
        return 0
   
    month_year = check_and_convert_int(month_year)
    if 11985 <= month_year <= 122021:
        return 1
    elif month_year == 777777 or 771980 <= month_year <= 772021:
        return 7
    elif month_year == 999999:
        return 9
    else:
        raise ValueError(invalid_col_val_msg('HIVTSTD3', month_year))
