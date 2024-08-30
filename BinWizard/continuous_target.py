from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
from optbinning import ContinuousOptimalBinning
from scipy import stats
from scipy.stats import somersd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_excel(df, filename="binning_results.xlsx"):
    """Generates an Excel file with the sample data and the assigned bins."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Binning Results')
    processed_data = output.getvalue()
    return processed_data

def download_button(data, filename, label):
    """Creates a button to download the data as an Excel file."""
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def interactive_binning(df):
    st.title("Interactive Binning")

    # Select numeric variables
    numeric_columns = df.select_dtypes(exclude=['object', 'category', 'datetime64[ns]', 'datetime64']).columns.tolist()
    if not numeric_columns:
        st.error("No numeric columns found in the dataset.")
        return

    continuous_var = st.selectbox("Select numerical variable for binning", numeric_columns)
    target_var = st.selectbox("Select target variable", numeric_columns)  # Now only allows numeric variables

    # Binning configuration
    max_n_bins = st.slider("Maximum number of bins", 2, 20, 10)
    min_bin_size = st.slider("Minimum bin size (%)", 0.0, 20.0, 5.0) / 100
    monotonic_trend = st.selectbox("Monotonic trend", ["auto", "ascending", "descending", "convex", "concave", "none"])

    # Perform initial binning for the continuous variable with the continuous target
    optb = ContinuousOptimalBinning(name=continuous_var,
                                    max_n_bins=max_n_bins,
                                    min_bin_size=min_bin_size,
                                    monotonic_trend=monotonic_trend)

    optb.fit(df[continuous_var], df[target_var])

    # Validate and clean the splits
    valid_splits = validate_splits(optb.splits)

    # Calculate the metrics using ContinuousOptimalBinning
    table = optb.binning_table.build()
    table = table[~table['Bin'].isin(['Special', 'Missing'])]
    table['Bin'] = table['Bin'].apply(lambda x: str(x))

    # Calculate Somers' D manually
    somers_d_value = calculate_somersd(df, continuous_var, target_var, valid_splits)

    # Display the table with the bins and Somers' D
    st.subheader("Binning Results (Automatic)")
    st.write(table)

    # Button to download the full sample with assigned bins
    excel_data = generate_excel(df)
    download_button(excel_data, "binning_results.xlsx", "Download full sample with bins")

    st.subheader("Somers' D (Automatic)")
    st.write(f"Somers' D: {somers_d_value:.4f}")

    # Visualization of violin plots with p-values for the automatic splits
    if valid_splits is not None and len(valid_splits) > 0:
        df['bin'] = pd.cut(df[continuous_var], bins=valid_splits, include_lowest=True)
        plot_violinplots_with_pvalues(df, continuous_var, target_var)

    # Manual split selection and combination
    st.subheader("Manual Split Combination")
    st.write(f"Initial Splits: {valid_splits[1:-1]}")  # Display splits without infinities

    if valid_splits is not None and len(valid_splits) > 0:
        selected_splits = st.multiselect("Select splits to remove", options=valid_splits[1:-1],
                                         default=valid_splits[1:-1])
        if st.button("Combine Selected Splits"):
            combined_splits = combine_splits(valid_splits, selected_splits)
            st.write(f"New Combined Splits: {combined_splits[1:-1]}")

            # Apply the combined splits to the data
            df['bin'] = pd.cut(df[continuous_var], bins=combined_splits, include_lowest=True)

            # Calculate and display the new result table with metrics similar to optbinning
            new_table = calculate_metrics_continuous(df, continuous_var, target_var, combined_splits)
            st.subheader("Updated Binning Results (Manual)")
            st.write(new_table)

            # Visualize the distribution with the new splits
            plot_violinplots_with_pvalues(df, continuous_var, target_var)

            # Button to download the updated sample with the new assigned bins
            excel_data = generate_excel(df)
            download_button(excel_data, "updated_binning_results.xlsx", "Download updated sample with bins")

def validate_splits(splits):
    """Validates and cleans the splits by removing duplicates and NaN."""
    if splits is None:
        return []
    splits = np.unique([split for split in splits if not pd.isna(split)])
    if len(splits) < 2:
        return []
    return [-np.inf] + splits.tolist() + [np.inf]

def combine_splits(splits, selected_splits):
    """Combines all splits, except the selected ones."""
    new_splits = [-np.inf]  # Always include the lower bound

    for split in splits[1:-1]:  # Ignore the first and last (infinities)
        if split in selected_splits:
            new_splits.append(split)

    new_splits.append(np.inf)  # Always include the upper bound
    return np.unique(new_splits)

def calculate_metrics_continuous(df, continuous_var, target_var, splits):
    """Calculates metrics equivalent to those of ContinuousOptimalBinning for a continuous target."""
    df['bin'] = pd.cut(df[continuous_var], bins=splits, include_lowest=True)
    labels = [f"({bin.left:.2f}, {bin.right:.2f}]" for bin in df['bin'].cat.categories]
    df['bin_label'] = pd.cut(df[continuous_var], bins=splits, include_lowest=True, labels=labels)

    # Calculate the mean and variance of the target within each bin
    table = df.groupby('bin_label').agg(
        Count=('bin_label', 'size'),
        Mean_Target=(target_var, 'mean'),
        Var_Target=(target_var, 'var')
    )

    # Calculate the HHI using the new calculate_herfindahl function
    hhi_result = calculate_herfindahl(table)
    table['HHI'] = hhi_result["HI"]  # Use HI_trad or HI as required

    # Calculate Somers' D manually
    somers_d_value = calculate_somersd(df, continuous_var, target_var, splits)
    table['SomersD'] = somers_d_value

    # Reset the index to display the bin labels clearly
    table = table.reset_index()

    return table

def calculate_herfindahl(table):
    counts = table['Count']
    total = counts.sum()

    # Calculate the frequencies
    concentration = counts.value_counts().reset_index()
    concentration.columns = ['HRC', 'Nobs']
    concentration['freq'] = concentration['Nobs'] / total

    # Number of categories (K)
    K = len(concentration)
    coef = 1 / K

    # Calculate the index
    concentration['index'] = (concentration['freq'] - coef) ** 2

    # Calculate CV
    CV2 = (np.sqrt(np.sum(concentration['index']) * K)) ** 2
    CV = np.sqrt(np.sum(concentration['index']) * K)

    # Calculate HI
    HI = 1 + np.log((CV2 + 1) / K) / np.log(K)

    # Calculate the traditional HI
    HI_trad = np.sum(concentration['freq'] ** 2)

    return {
        "HI": HI,
        "CV": CV,
        "HI_trad": HI_trad,
        "table": concentration
    }

def calculate_somersd(df, continuous_var, target_var, splits):
    df['bin'] = pd.cut(df[continuous_var], bins=splits, include_lowest=True)
    df['bin_code'] = pd.Categorical(df['bin']).codes
    somers_d_result = somersd(df['bin_code'], df[target_var])
    return somers_d_result.statistic  # Access the value of Somers' D

def plot_violinplots_with_pvalues(df, continuous_var, target_var):
    bins = df['bin'].cat.categories
    p_values = []

    for i in range(len(bins) - 1):
        bin1_data = df[df['bin'] == bins[i]][target_var]
        bin2_data = df[df['bin'] == bins[i + 1]][target_var]
        if len(bin1_data) > 0 and len(bin2_data) > 0:
            # Apply the Wilcoxon Rank-Sum test (Mann-Whitney U)
            stat, p_value = stats.ranksums(bin1_data, bin2_data)
            p_values.append(p_value)
        else:
            p_values.append(np.nan)

    # Create the violin plot
    fig, ax = plt.subplots(figsize=(6, 2))
    sns.violinplot(x='bin', y=continuous_var, data=df, ax=ax, inner='quartile', palette='muted')

    # Add p-values between adjacent bins
    for i, p_value in enumerate(p_values):
        x1, x2 = i, i + 1
        y, h, col = df[continuous_var].max() + df[continuous_var].std() / 2, df[continuous_var].std() / 10, 'k'
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        formatted_p_value = f"p={p_value:.4f}"  # Round to 4 decimal places
        ax.text((x1 + x2) * 0.5, y + h, formatted_p_value, ha='center', va='bottom', color=col, fontsize=8)

    ax.set_title(f"Violin Plots of {continuous_var} with Wilcoxon p-values", fontsize=8)
    ax.set_xlabel("Bins", fontsize=8)
    ax.set_ylabel(continuous_var, fontsize=8)
    sns.despine()

    plt.xticks(rotation=30, ha='right', fontsize=5)  # Rotate x labels if necessary
    st.pyplot(fig)

def main():
    st.set_page_config(layout="wide")

    # File uploader that accepts multiple file formats
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "sas7bdat"])

    if uploaded_file is not None:
        # Detect file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.sas7bdat'):
            df = pd.read_sas(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return

        interactive_binning(df)
    else:
        st.write("Please upload a CSV, Excel, or SAS file to begin.")

if __name__ == "__main__":
    main()
