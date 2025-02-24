import matplotlib

matplotlib.use('TkAgg')
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def filter_results(res_df, graph_layer=None, embedding_size=None, subgraph_size=None, node_features=None,
                   prediction_layer=None):
    filter_to_use = True
    if graph_layer is not None:
        filter_to_use = filter_to_use & (res_df['graph_layer'] == graph_layer)
    if embedding_size is not None:
        filter_to_use = filter_to_use & (res_df['embedding_size'] == embedding_size)
    if subgraph_size is not None:
        filter_to_use = filter_to_use & (res_df['subgraph_size'] == subgraph_size)
    if node_features is not None:
        filter_to_use = filter_to_use & (res_df['node_features'] == node_features)
    if prediction_layer is not None:
        filter_to_use = filter_to_use & (res_df['prediction_layer'] == prediction_layer)
    return res_df.loc[np.where(filter_to_use)]


def results_to_pd(root_dir, prediction_task):
    # Initialize a dataframe to collect all results
    all_results = []

    # Walk through the directory and read CSV files
    for subdir, _, files in os.walk(os.path.join(root_dir, prediction_task)):
        for file in files:
            if file.endswith(".csv"):
                parts = file.split('_')
                graph_layer, embedding_size, node_features = parts[0], int(parts[1]), parts[2].replace('.csv', '')
                subgraph_size = int(os.path.basename(subdir))

                # Read the entire CSV file and drop empty rows
                df = pd.read_csv(os.path.join(subdir, file)).dropna()

                if prediction_task == "remaining_time":
                    # Select the best score (minimum) for each prediction layer
                    best_results = df.groupby('prediction_layer', as_index=False)['score'].max()
                else:
                    # Select the best score (minimum) for each prediction layer
                    best_results = df.groupby('prediction_layer', as_index=False)['score'].min()
                # Collect results
                for _, row in best_results.iterrows():
                    if "models" in row['prediction_layer']:
                        all_results.append({
                            'subgraph_size': subgraph_size,
                            'graph_layer': graph_layer,
                            'embedding_size': int(embedding_size),
                            'node_features': node_features,
                            'prediction_layer': row['prediction_layer'],
                            'score': float(row['score'])
                        })

    # Convert to DataFrame for easier plotting
    results_df = pd.DataFrame(all_results)
    return results_df


def plot_subplots(results_df, prediction_task, group_by="graph_layer", x_axis_var="subgraph_size",
                  save_path=None):
    # Grouping and aggregating scores based on the specified attribute
    group_cols_emb = [group_by, "embedding_size"]
    group_cols_sub = [group_by, "subgraph_size"]

    agg_emb = results_df.groupby(group_cols_emb, as_index=False).agg({"score": 'mean'})
    #agg_emb['prediction_layer'] = 'Aggregated'
    agg_emb['subgraph_size'] = 'Aggregated'

    agg_sub = results_df.groupby(group_cols_sub, as_index=False).agg({"score": 'mean'})
    #agg_sub['prediction_layer'] = 'Aggregated'
    agg_sub['embedding_size'] = 'Aggregated'

    # Choose the data based on the x-axis variable
    if x_axis_var == "subgraph_size":
        data_to_plot = agg_sub
        xlabel = "Subgraph Size"
    elif x_axis_var == "embedding_size":
        data_to_plot = agg_emb
        xlabel = "Embedding Size"
    else:
        raise ValueError("Invalid x_axis_var. Choose 'subgraph_size' or 'embedding_size'.")

    # Create subplots with shared y-axis and a common legend
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), dpi=300, sharex=False, sharey=True)

    # Line plot: Selected attribute vs X-axis variable
    sns.lineplot(data=data_to_plot,x=x_axis_var,y='score',hue=group_by,marker='o',ax=axs[0])
    axs[0].legend_.remove()

    axs[0].set_title(prediction_task)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel("MAE (days)" if prediction_task == "remaining_time" else "Accuracy")
    axs[0].grid(True)

    # Bar plot: Selected attribute vs Embedding Size
    sns.barplot(ax=axs[1], data=agg_emb, x="embedding_size", y="score", hue=group_by, dodge=True, width=0.4,
                palette="muted")
    axs[1].legend_.remove()

    # Adjust y-axis range based on score type
    min_score = results_df["score"].min()
    max_score = results_df["score"].max()
    if prediction_task == "remaining_time":
        axs[1].set_ylim(min_score * 0.99, max_score * 1.01)
    else:
        axs[1].set_ylim(min_score - 0.1, max_score + 0.1)

    axs[1].set_title(f"Performance by Embedding Size ({group_by})")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("MAE (days)" if prediction_task == "remaining_time" else "Accuracy")
    axs[1].grid(axis='y')

    # Create a common y-axis label for both subplots
    # fig.text(0.04, 0.5, "Score (Error)" if score != "acc" else "Score (Accuracy)", va='center', rotation='vertical',
    #         fontsize=12)

    # Add a single common legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=group_by, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels))

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_multiple_subplots(results_df, prediction_task, save_path=None):
    # Define grouping options for line and bar plots
    grouping_options = [
        ("graph_layer", "subgraph_size", "Subgraph Size"),
        ("graph_layer", "embedding_size", "Embedding Size"),
        ("prediction_layer", "subgraph_size", "Subgraph Size"),
        ("prediction_layer", "embedding_size", "Embedding Size")
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=300, sharey=True)

    # Define separate color palettes for each row
    palette_row1 = sns.color_palette("muted", n_colors=len(results_df["graph_layer"].unique()))
    palette_row2 = sns.color_palette("bright", n_colors=len(results_df["prediction_layer"].unique()))

    for idx, (group_by, x_axis_var, xlabel) in enumerate(grouping_options):
        grouped_df = results_df.groupby([group_by, x_axis_var], as_index=False).agg({"score": 'mean'})

        row = idx // 2
        col = idx % 2

        # Select palette based on row (0: graph_layer, 1: prediction_layer)
        palette = palette_row1 if row == 0 else palette_row2

        if idx % 2 == 0:
            # Line plot for trend visualization
            for i, group in enumerate(grouped_df[group_by].unique()):
                subset = grouped_df[grouped_df[group_by] == group].sort_values(x_axis_var)
                axs[row, col].plot(subset[x_axis_var], subset['score'], marker='o', label=group, color=palette[i])
        else:
            # Bar plot for grouped visualization
            sns.barplot(ax=axs[row, col], data=grouped_df, x=x_axis_var, y="score", hue=group_by, dodge=True, width=0.4,
                        palette=palette)
            axs[row, col].legend_.remove()  # Remove duplicate legends from barplots

        # Plot formatting
        axs[row, col].set_title(f"{prediction_task} ({group_by} vs {xlabel})")
        axs[row, col].set_xlabel(xlabel)
        axs[row, col].grid(True)

    # Set a common Y-axis scale
    min_score = results_df["score"].min()
    max_score = results_df["score"].max()
    for ax in axs.flat:
        if prediction_task == "remaining_time":
            ax.set_ylim(min_score * 0.99, max_score * 1.01)
        else:
            ax.set_ylim(min_score - 0.1, max_score + 0.1)

    # Common Y-axis label
    # fig.text(0.04, 0.5, "MAE (in days)" if score != "acc" else "Accuracy", va='left', rotation='vertical',
    #         fontsize=12)

    # Adjust layout to make space for legends
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right

    # Add shared legends outside the plot area
    handles_row1, labels_row1 = axs[0, 0].get_legend_handles_labels()
    handles_row2, labels_row2 = axs[1, 0].get_legend_handles_labels()

    fig.legend(handles_row1, labels_row1, title="Graph Layers", loc='center left', bbox_to_anchor=(1, 0.75))
    fig.legend(handles_row2, labels_row2, title="Prediction Layers", loc='center left', bbox_to_anchor=(1, 0.25))

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def compute_weighted_avg(root_dir, prediction_task, embedding_size, weights_mapping, dataset_label):
    # Load results using your existing function
    results_df = results_to_pd(root_dir, prediction_task)
    # Filter by the desired embedding size (e.g., 32)
    res = results_df.loc[results_df['embedding_size'] == embedding_size].copy()
    # Map each subgraph_size to its corresponding weight
    res['weights'] = res['subgraph_size'].map(weights_mapping)
    # Compute the weighted average score for each (graph_layer, prediction_layer)
    weighted_avg = (
        res.groupby(['graph_layer', 'prediction_layer'])
           .apply(lambda group: (group['score'] * group['weights']).sum() / group['weights'].sum())
           .reset_index(name=dataset_label)
    )
    return weighted_avg


