from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
from optbinning import OptimalBinning
from scipy import stats
from scipy.stats import somersd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


def generate_excel(df, filename="binning_results.xlsx"):
    """Generates an Excel file with the sample data and assigned bins."""
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
    target_var = st.selectbox("Select target variable", df.columns)

    # Binning configuration
    max_n_bins = st.slider("Maximum number of bins", 2, 20, 10)
    min_bin_size = st.slider("Minimum bin size (%)", 0.0, 20.0, 5.0) / 100
    monotonic_trend = st.selectbox("Monotonic trend", ["auto", "ascending", "descending", "convex", "concave", "none"])

    # Perform initial binning
    optb = OptimalBinning(name=continuous_var,
                          dtype="numerical",
                          solver="cp",
                          max_n_bins=max_n_bins,
                          min_bin_size=min_bin_size,
                          monotonic_trend=monotonic_trend)

    optb.fit(df[continuous_var], df[target_var])

    # Validate and clean the splits
    valid_splits = validate_splits(optb.splits)

    # Calculate HHI and Somers' D for automatic splits
    table = optb.binning_table.build()
    table = table[~table['Bin'].isin(['Special', 'Missing'])]
    table['Bin'] = table['Bin'].apply(lambda x: str(x))
    hhi_value = calculate_herfindahl(table)

    if valid_splits is not None and len(valid_splits) > 0:
        somers_d_value = calculate_somersd(df, continuous_var, target_var, valid_splits)
    else:
        somers_d_value = np.nan
        st.warning("No valid splits found. Somers' D could not be calculated.")

    # Display the table with bins, HHI, and automatic Somers' D
    st.subheader("Binning Results (Automatic)")
    st.write(table)

    # Button to download the full sample with assigned bins
    excel_data = generate_excel(df)
    download_button(excel_data, "binning_results.xlsx", "Download full sample with bins")

    st.subheader("HHI and Somers' D (Automatic)")
    hhi_value = calculate_herfindahl(table)["HI"]
    st.write(f"Herfindahl-Hirschman Index (HHI): {hhi_value:.4f}")
    st.write(f"Somers' D: {somers_d_value:.4f}")

    # Visualization of violin plots with p-values for automatic splits
    if valid_splits is not None and len(valid_splits) > 0:
        df['bin'] = pd.cut(df[continuous_var], bins=valid_splits, include_lowest=True)
        plot_violinplots_with_pvalues_matrix(df, continuous_var, target_var)

    # Manual split selection and combination
    st.subheader("Manual Split Combination")
    st.write(f"Initial Splits: {valid_splits[1:-1]}")  # Display splits without infinities

    if valid_splits is not None and len(valid_splits) > 0:
        selected_splits = st.multiselect("Select splits to remove", options=valid_splits[1:-1],
                                         default=valid_splits[1:-1])
        if st.button("Combine Selected Splits"):
            combined_splits = combine_splits(valid_splits, selected_splits)
            st.write(f"New Combined Splits: {combined_splits[1:-1]}")

            # Apply combined splits to the data
            df['bin'] = pd.cut(df[continuous_var], bins=combined_splits, include_lowest=True)

            # Calculate and display the new result table
            new_table = calculate_metrics(df, continuous_var, target_var, combined_splits)
            st.subheader("Updated Binning Results (Manual)")
            st.write(new_table)

            # Visualize distribution with the new splits
            plot_violinplots_with_pvalues_matrix(df, continuous_var, target_var)

            # Button to download the updated sample with assigned bins
            excel_data = generate_excel(df)
            download_button(excel_data, "updated_binning_results.xlsx", "Download updated sample with bins")

    # Manual split creation section
    st.subheader("Manual Split Creation")

    manual_splits_input = st.text_input("Enter manual splits separated by commas (e.g., 0.1, 0.5, 0.9)")
    if manual_splits_input:
        try:
            # Convert input to a sorted list of floats
            manual_splits = sorted([float(x.strip()) for x in manual_splits_input.split(',')])
            manual_splits = [-np.inf] + manual_splits + [np.inf]
            st.write(f"Using manual splits: {manual_splits[1:-1]}")

            # Apply manual splits to the data
            df['bin'] = pd.cut(df[continuous_var], bins=manual_splits, include_lowest=True)

            # Calculate and display the result table
            manual_table = calculate_metrics(df, continuous_var, target_var, manual_splits)
            st.subheader("Manual Binning Results")
            st.write(manual_table)

            # Visualize distribution with the manual splits
            plot_violinplots_with_pvalues_matrix(df, continuous_var, target_var)

            # Button to download the manually binned sample
            excel_data = generate_excel(df)
            download_button(excel_data, "manual_binning_results.xlsx", "Download manually binned sample with bins")

        except ValueError:
            st.error("Invalid input for manual splits. Please enter numbers separated by commas.")

def plot_violinplots_with_pvalues_matrix(df, continuous_var, target_var):
    # Filter out invalid bins
    df = df[df['bin_code'] != -1]

    # Ensure bins are ordered correctly
    bins = df['bin'].cat.categories
    df['bin'] = pd.Categorical(df['bin'], categories=bins, ordered=True)

    # Create an empty matrix to store p-values
    p_values_matrix = np.full((len(bins), len(bins)), np.nan)

    # Calculate p-values for all pairs of bins
    for i, j in itertools.combinations(range(len(bins)), 2):
        bin1_data = df[df['bin'] == bins[i]][target_var]
        bin2_data = df[df['bin'] == bins[j]][target_var]
        if len(bin1_data) > 0 and len(bin2_data) > 0:
            _, p_value = stats.ranksums(bin1_data, bin2_data)
            p_values_matrix[i, j] = p_value
            p_values_matrix[j, i] = p_value  # Symmetric matrix

    # Convert the p-values matrix into a DataFrame for better display
    p_values_df = pd.DataFrame(p_values_matrix, index=bins, columns=bins)

    # Display the matrix using Streamlit
    st.subheader("P-Value Matrix for Wilcoxon Rank-Sum Test Between Bins")
    st.dataframe(p_values_df.style.format(precision=4))


def validate_splits(splits):
    """Validates and cleans the splits by removing duplicates and NaN."""
    if splits is None:
        return []
    splits = np.unique([split for split in splits if not pd.isna(split)])
    if len(splits) < 2:
        return []
    return [-np.inf] + splits.tolist() + [np.inf]


def combine_splits(splits, selected_splits):
    """Combines all splits except the selected ones."""
    new_splits = [-np.inf]  # Always include the lower bound

    for split in splits[1:-1]:  # Ignore the first and last (infinities)
        if split in selected_splits:
            new_splits.append(split)

    new_splits.append(np.inf)  # Always include the upper bound
    return np.unique(new_splits)


def calculate_metrics(df, continuous_var, target_var, splits):
    """Calculates HHI, Somers' D, and returns a table with results, including labels for the new bins."""
    # Create labeled bins
    df['bin'] = pd.cut(df[continuous_var], bins=splits, include_lowest=True)
    df = df.dropna(subset=['bin'])
    df['bin_code'] = pd.Categorical(df['bin']).codes
    df = df[df['bin_code'] != -1]
    df = df.sort_values(by='bin_code')
    labels = [f"({bin.left:.2f}, {bin.right:.2f}]" for bin in df['bin'].cat.categories]
    df['bin_label'] = pd.cut(df[continuous_var], bins=splits, include_lowest=True, labels=labels)

    # Group by the labeled bins
    table = df.groupby('bin_label').agg(
        Count=('bin_label', 'size'),
        Non_event=(target_var, lambda x: (x == 0).sum()),
        Event=(target_var, lambda x: (x == 1).sum())
    )
    table['Event rate'] = table['Event'] / table['Count']
    table['WoE'] = np.log((table['Event'] / table['Event'].sum()) / (table['Non_event'] / table['Non_event'].sum()))
    table['IV'] = (table['Event'] / table['Event'].sum() - table['Non_event'] / table['Non_event'].sum()) * table['WoE']

    # Calculate HHI using the new calculate_herfindahl function
    hhi_result = calculate_herfindahl(table)
    table['HHI'] = hhi_result["HI"]  # Use HI_trad or HI as required
    somers_d_value = calculate_somersd(df, continuous_var, target_var, splits)
    table['SomersD'] = somers_d_value
    table['IV'] = table['IV'].sum()

    # Reset the index to display the bin labels clearly
    table = table.reset_index()

    return table


def calculate_herfindahl(table):
    counts = table['Count']
    total = counts.sum()

    # Calculate frequencies
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
    df = df[df['bin_code'] != -1]
    df = df.sort_values(by='bin_code')
    somers_d_result = somersd(df[target_var], df['bin_code'])
    return somers_d_result.statistic  # Access the Somers' D value


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
