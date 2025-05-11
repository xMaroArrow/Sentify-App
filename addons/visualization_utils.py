# visualization_utils.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def create_sentiment_pie_chart(sentiment_counts):
    """Create sentiment distribution pie chart with dark theme."""
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    
    # Set dark theme
    plt.style.use('dark_background')
    
    # Create colors
    colors = ["#4CAF50", "#FFC107", "#F44336"]  # Green, Amber, Red
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sentiment_counts.values(),
        labels=sentiment_counts.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=(0.05, 0, 0),  # Slightly emphasize positive
        wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
    )
    
    # Enhance text styling
    for text in texts:
        text.set_color('white')
        text.set_fontweight('bold')
        
    for autotext in autotexts:
        text.set_color('white')
        text.set_fontsize(8)
        
    # Add title
    ax.set_title("Sentiment Distribution", color='white', fontsize=14)
    
    # Add count info
    total = sum(sentiment_counts.values())
    ax.text(
        -0.1, -1.2,
        f"Total tweets: {total}",
        color='white',
        fontsize=10
    )
    
    return fig, ax

def create_trend_chart(sentiment_trend):
    """Create time trend visualization with dark theme."""
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    
    # Set dark theme
    plt.style.use('dark_background')
    
    # Plot lines for each sentiment
    sentiment_trend.plot(
        ax=ax,
        color=["#4CAF50", "#FFC107", "#F44336"],  # Green, Amber, Red
        linewidth=2,
        marker='o',
        markersize=4
    )
    
    # Set labels and title
    ax.set_title("Sentiment Trends Over Time", color='white', fontsize=14)
    ax.set_xlabel("Time", color='white')
    ax.set_ylabel("Tweet Count", color='white')
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend
    ax.legend(title="Sentiment")
    
    return fig, ax

def create_language_visualization(language_distribution, language_names):
    """Create language visualization with dark theme."""
    # Keep only top languages for readability
    top_languages = language_distribution.head(8)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(8, 5), dpi=100)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
    
    # Set dark theme
    plt.style.use('dark_background')
    
    # Left subplot: Language pie chart
    ax1 = fig.add_subplot(gs[0])
    
    # Get language counts
    langs = [language_names.get(l, l) for l in top_languages['language']]
    counts = top_languages['total']
    
    # Create custom colormap for languages
    colors = plt.cm.tab10(np.linspace(0, 1, len(langs)))
    
    # Create pie chart
    wedges, texts = ax1.pie(
        counts,
        labels=None,  # No labels on pie chart
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
    )
    
    # Add legend
    ax1.legend(
        wedges,
        [f"{l} ({c})" for l, c in zip(langs, counts)],
        loc="center left",
        bbox_to_anchor=(0, 0.5),
        fontsize=8
    )
    
    ax1.set_title("Language Distribution", color='white', fontsize=12)
    
    # Right subplot: Sentiment by language (top 5)
    ax2 = fig.add_subplot(gs[1])
    
    # Filter to top languages
    top_lang_codes = top_languages['language'].tolist()[:5]
    
    # Create data for stacked bar chart
    data = []
    for lang in top_lang_codes:
        row = top_languages[top_languages['language'] == lang].iloc[0]
        lang_name = language_names.get(lang, lang)
        data.append({
            "language": lang_name,
            "Positive": row["Positive"],
            "Neutral": row["Neutral"],
            "Negative": row["Negative"]
        })
    
    # Set up bar chart
    langs = [d["language"] for d in data]
    pos = np.arange(len(langs))
    bar_width = 0.6
    
    # Plot stacked bars
    ax2.bar(
        pos, [d["Positive"] for d in data], bar_width,
        color="#4CAF50", label="Positive", edgecolor='white', linewidth=0.5
    )
    
    ax2.bar(
        pos, [d["Neutral"] for d in data], bar_width,
        bottom=[d["Positive"] for d in data],
        color="#FFC107", label="Neutral", edgecolor='white', linewidth=0.5
    )
    
    # Calculate bottom position for negative sentiment
    bottoms = [d["Positive"] + d["Neutral"] for d in data]
    
    ax2.bar(
        pos, [d["Negative"] for d in data], bar_width,
        bottom=bottoms,
        color="#F44336", label="Negative", edgecolor='white', linewidth=0.5
    )
    
    # Set labels and title
    ax2.set_title("Sentiment by Language", color='white', fontsize=12)
    ax2.set_xlabel("Language", color='white')
    ax2.set_ylabel("Tweet Count", color='white')
    
    # Set x-tick labels
    ax2.set_xticks(pos)
    ax2.set_xticklabels(langs, rotation=45, ha="right")
    
    # Add legend and grid
    ax2.legend(title="Sentiment")
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig