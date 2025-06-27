import shap             
import pandas as pd      
import matplotlib.pyplot as plt  


def plot_shap_bar_side_by_side(expl1, expl2, 
                         title1="Group 1", title2="Group 2", 
                         max_display=20, figsize=(15, 8)):
    """
    Plot side-by-side SHAP bar plots for two explanation objects.
    
    Parameters:
        expl1 (shap.Explanation): SHAP explanation for the first group.
        expl2 (shap.Explanation): SHAP explanation for the second group.
        title1 (str): Title for the first plot.
        title2 (str): Title for the second plot.
        max_display (int): Max features to display.
        figsize (tuple): Figure size.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    shap.plots.bar(expl1, max_display=max_display, ax=ax1, show=False)
    ax1.set_title(title1)

    shap.plots.bar(expl2, max_display=max_display, ax=ax2, show=False)
    ax2.set_title(title2)

    plt.tight_layout()
    plt.show()


def get_shap_comparison_df(group1_mean, group2_mean, feature_names, 
                           group1_label="Group_1", group2_label="Group_2"):
    """
    Create a DataFrame comparing mean SHAP values of two groups.

    Parameters:
        group1_mean (array-like): SHAP mean values for group 1.
        group2_mean (array-like): SHAP mean values for group 2.
        feature_names (list): Feature names corresponding to SHAP values.
        group1_label (str): Label for group 1.
        group2_label (str): Label for group 2.

    Returns:
        pd.DataFrame: Sorted DataFrame with feature-wise SHAP differences.
    """
    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        group1_label: group1_mean,
        group2_label: group2_mean,
        'Difference': group1_mean - group2_mean
    }).sort_values('Difference', key=abs, ascending=False)
    
    return comparison_df


def plot_shap_difference_bar(group1_mean, group2_mean, feature_names, 
                             group1_label="Group 1", group2_label="Group 2",
                             top_n=15, figsize=(12, 8)):
    """
    Plot top SHAP value differences between two groups (e.g., FN vs TN, FP vs TP).

    Parameters:
        group1_mean (array-like): SHAP mean values for group 1.
        group2_mean (array-like): SHAP mean values for group 2.
        feature_names (list): List of feature names corresponding to SHAP values.
        group1_label (str): Label for group 1.
        group2_label (str): Label for group 2.
        top_n (int): Number of top features to display.
        figsize (tuple): Size of the figure.
    """

    comparison_df = get_shap_comparison_df(
        group1_mean, group2_mean, feature_names,
        group1_label=group1_label, group2_label=group2_label
    )

    top_features = comparison_df.head(top_n)

    # Plot
    plt.figure(figsize=figsize)
    plt.barh(
        range(len(top_features)),
        top_features['Difference'],
        color=['red' if x > 0 else 'blue' for x in top_features['Difference']]
    )
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel(f'SHAP Value Difference ({group1_label} - {group2_label})')
    plt.title(f'Feature Importance Differences: {group1_label} vs {group2_label}')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_shap_violin_comparison(expl1, expl2, feature_names,
                                comparison_df, 
                                top_n=10,
                                group1_label="Group 1",
                                group2_label="Group 2",
                                figsize=(20, 10)):
    """
    Plot violin plots comparing SHAP value distributions of top N different features
    between two explanation groups (e.g., FN vs TN or FP vs TP).

    Parameters:
        expl1 (shap.Explanation): SHAP explanation for group 1.
        expl2 (shap.Explanation): SHAP explanation for group 2.
        feature_names (list): List of all feature names.
        comparison_df (pd.DataFrame): DataFrame with 'Feature' and 'Difference' columns.
        top_n (int): Number of top features to visualize.
        group1_label (str): Label for group 1 (used in plot).
        group2_label (str): Label for group 2.
        figsize (tuple): Size of the overall figure.
    """
    top_features = comparison_df.head(top_n)['Feature'].tolist()

    fig, axes = plt.subplots(2, top_n // 2, figsize=figsize)
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        feature_idx = feature_names.index(feature)
        values1 = expl1.values[:, feature_idx]
        values2 = expl2.values[:, feature_idx]

        data_to_plot = [values1, values2]

        axes[i].violinplot(data_to_plot, positions=[1, 2])
        axes[i].set_xticks([1, 2])
        axes[i].set_xticklabels([group1_label, group2_label])
        axes[i].set_title(feature)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f'SHAP Value Distributions: {group1_label} vs {group2_label}', y=1.02)
    plt.show()


