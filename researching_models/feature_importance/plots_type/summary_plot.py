import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

def create_summary_plot(shap_values, X_sample, feature_names, plot_data, plot_type, sort_order, task_type, explainer_type, is_multi_output=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Используем shap_values напрямую как explanation, так как он уже содержит правильные feature_names
    explanation = shap_values
    
    shap.plots.beeswarm(explanation, show=False)
    
    ax = plt.gca()
    ax.set_title(f"{plot_type} - {sort_order}")
    current_fig = plt.gcf()
    return current_fig
