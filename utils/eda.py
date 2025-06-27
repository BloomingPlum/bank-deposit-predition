import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

def bi_countplot_target(df, column, hue_column='y', top_n=20, figsize=(16, 6)):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # Only take top N values to reduce overcrowding
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True) * 100
    top_values = proportions.groupby(level=1).sum().sort_values(ascending=False).head(top_n).index

    # Filter to only include top values
    filtered_df = df[df[column].isin(top_values)]

    # Left plot - Normalized distribution
    pltname = f'Normalized Distribution by Category': {column}'
    proportions = filtered_df.groupby(hue_column)[column].value_counts(normalize=True) * 100
    ax = proportions.unstack(hue_column).plot.bar(ax=axes[0], title=pltname)

    # Add labels selectively to avoid overcrowding
    for i, container in enumerate(ax.containers):
        if i < 2:  # Only label first two categories
            ax.bar_label(container, fmt='{:,.1f}%', fontsize=8)

    # Right plot - Count distribution
    pltname = f'Number of Values by Category': {column}'
    counts = filtered_df.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).plot.bar(ax=axes[1], title=pltname)

    # Add labels selectively
    for i, container in enumerate(ax.containers):
        if i < 2:  # Only label first two categories
            ax.bar_label(container, fontsize=8)

    # Improve readability
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(title=hue_column)

    plt.tight_layout()
    plt.show()


def analyze_responses(df, feature_col, response_col="y"):
    """
    Analyze response data by grouping by a feature column and calculating counts and percentages.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to analyze
    feature_col : str
        The column to group by (e.g., 'previous', 'pdays', 'campaign')
    response_col : str, default="y"
        The column containing binary responses (typically "yes"/"no")
    
    Returns:
    --------
    pandas.DataFrame
        A dataframe with counts and percentages for each response type
    """
    # Group by the feature column and response column, then count occurrences
    response = df.groupby([feature_col, response_col]).size().unstack(fill_value=0)
    
    # Rename the columns
    response.columns = ["no_count", "yes_count"]
    
    # Calculate percentages
    total = response["no_count"] + response["yes_count"]
    response["no_percentage"] = (response["no_count"] / total * 100).round(2)
    response["yes_percentage"] = (response["yes_count"] / total * 100).round(2)
    
    # Reset index to make the grouped column a regular column
    response = response.reset_index()
    
    return response


def prepare_dates_and_plot(df, real_months_order=None, show_plot=False):
    """
    Prepares a DataFrame with chronological dates based on month names and day of week,
    then plots positive and negative outcomes by month-year.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'month', 'day_of_week', and 'y' columns.
        'month' should be lowercase three-letter month names (e.g., 'jan', 'feb').
        'day_of_week' should be lowercase three-letter day names (e.g., 'mon', 'tue').
        'y' should contain binary responses (e.g., 'yes', 'no').
    
    real_months_order : list of tuples, optional
        List of (month_str, year_num) in chronological order.
        If None, uses default order from May 2008 to Nov 2010.
    
    Returns:
    --------
    tuple
        (processed_df, counts_per_month, matplotlib_axis)
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # 1) Ensure DataFrame is sorted and reset indices
    df.sort_index(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 2) Define real months order if not provided
    if real_months_order is None:
        real_months_order = [
            # 2008
            ("may", 2008), ("jun", 2008), ("jul", 2008), ("aug", 2008),
            ("sep", 2008), ("oct", 2008), ("nov", 2008), ("dec", 2008),
            # 2009
            ("jan", 2009), ("feb", 2009), ("mar", 2009), ("apr", 2009),
            ("may", 2009), ("jun", 2009), ("jul", 2009), ("aug", 2009),
            ("sep", 2009), ("oct", 2009), ("nov", 2009), ("dec", 2009),
            # 2010
            ("jan", 2010), ("feb", 2010), ("mar", 2010), ("apr", 2010),
            ("may", 2010), ("jun", 2010), ("jul", 2010), ("aug", 2010),
            ("sep", 2010), ("oct", 2010), ("nov", 2010)
        ]
    
    # 3) Create an empty "year" column and populate it
    df["year"] = None
    idx_rm = 0  # Index in real_months_order
    n = len(real_months_order)
    
    for i in range(len(df)):
        row_month = df.loc[i, "month"]  # e.g. "mar", "may", etc.
        # Match the DataFrame's month to the correct item in real_months_order
        while idx_rm < n and real_months_order[idx_rm][0] != row_month:
            idx_rm += 1
        # If we overshoot the list, cap at the last item
        if idx_rm >= n:
            idx_rm = n - 1
        # Assign the year
        df.loc[i, "year"] = real_months_order[idx_rm][1]
    
    # 4) Build actual dates
    month_map = {
        "jan": 1,  "feb": 2,  "mar": 3,  "apr": 4,
        "may": 5,  "jun": 6,  "jul": 7,  "aug": 8,
        "sep": 9,  "oct": 10, "nov": 11, "dec": 12
    }
    days_map = {
        "mon": 0, "tue": 1, "wed": 2,
        "thu": 3, "fri": 4, "sat": 5,
        "sun": 6
    }
    
    df["true_date"] = pd.NaT
    current_date = None
    
    for i in range(len(df)):
        row_m = df.loc[i, "month"]       # "mar", etc.
        row_y = df.loc[i, "year"]        # from the new 'year' column
        row_d = df.loc[i, "day_of_week"] # "mon", "tue", ...
        
        # For each new month/year, start from the 1st day of that month
        if i == 0 or df.loc[i, "month"] != df.loc[i-1, "month"] or df.loc[i, "year"] != df.loc[i-1, "year"]:
            current_date = dt.date(row_y, month_map[row_m], 1)
            while current_date.weekday() != days_map[row_d]:
                current_date += dt.timedelta(days=1)
        else:
            # If month/year hasn't changed, adjust date if day_of_week is different
            prev_dow = df.loc[i-1, "day_of_week"]
            if row_d != prev_dow:
                next_date = current_date + dt.timedelta(days=1)
                while next_date.weekday() != days_map[row_d]:
                    next_date += dt.timedelta(days=1)
                current_date = next_date
        
        df.loc[i, "true_date"] = pd.to_datetime(current_date)
    
    # Create month-year period
    df["true_month_year"] = df["true_date"].dt.to_period("M")
    
    # Count 'yes'/'no' responses per month
    counts_per_month = (
        df.groupby("true_month_year")["y"]
        .value_counts()
        .unstack(fill_value=0)
    )
    
    # Rename columns to Ukrainian
    counts_per_month.rename(columns={"yes": "Positive", "no": "Negative"}, inplace=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 8))
    counts_per_month.plot(
        kind="bar",
        width=0.8,
        ax=ax
    )
    ax.set_xlabel("Month-Year")
    ax.set_ylabel("Number of Observations")
    ax.set_title("Number of Positive and Negative Responses by Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # This prevents automatic display
    
    return df, counts_per_month, ax


def plot_cramers_v_heatmap(df, target_col='y', target_mapping=None, figsize=(12, 10), cmap="YlGnBu"):
    """
    Computes pairwise Cramér's V for all categorical columns (including the target)
    and plots a heatmap.

    Parameters:
      df : pandas.DataFrame
          DataFrame containing your data.
      target_col : str, optional
          Name of the target column to be converted to categorical (default is 'y').
      target_mapping : dict, optional
          Mapping for the target column (e.g., {0: "no", 1: "yes"}). If provided,
          the target column is mapped using this dictionary before converting to object.
      figsize : tuple, optional
          Figure size for the heatmap plot.
      cmap : str, optional
          Colormap to be used in the heatmap.

    Returns:
      cramer_matrix : pandas.DataFrame
          DataFrame containing the pairwise Cramér's V values.
    """
    
    # Convert the target column to categorical using the provided mapping if available
    if target_mapping is not None:
        df[target_col] = df[target_col].map(target_mapping).astype("object")
    else:
        df[target_col] = df[target_col].astype("object")
    
    # Optionally create a temporary column (if needed) or use the converted target column directly.
    # Here we assume target_col is now categorical.
    
    # Select all categorical columns (object dtype)
    cat_cols = df.select_dtypes(include="object").columns
    
    # Define function to compute Cramér's V
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        return np.sqrt(chi2 / (n * (min(k, r) - 1)))
    
    # Compute the pairwise Cramér’s V matrix
    cramer_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)
    for col1 in cat_cols:
        for col2 in cat_cols:
            cramer_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
    cramer_matrix = cramer_matrix.astype(float)
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(cramer_matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5)
    plt.title("Cramér’s V Heatmap (Categorical Features)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return cramer_matrix