"""
Visualization Module.
Generates interactive Plotly charts for Sentiment Probabilities,
Emotion Distribution, TF-IDF feature importance, and Linguistic Metrics.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px

# Theme styling constants
PLOTLY_TEMPLATE = "plotly_white"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def create_sentiment_chart(sentiment_data: Dict[str, Any]) -> go.Figure:
    """
    Creates an interactive bar chart displaying 3-class Sentiment Probabilities.
    """
    probabilities = sentiment_data.get('probabilities_pct', {'Positive': 0, 'Neutral': 0, 'Negative': 0})
    
    categories = ['Positive', 'Neutral', 'Negative']
    scores = [probabilities.get(cat, 0.0) for cat in categories]
    colors = ['#10B981', '#6B7280', '#EF4444'] # Green, Grey, Red

    fig = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=categories,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255, 255, 255, 0.6)', width=1.5),
                    cornerradius=6
                ),
                text=[f"<b>{score:.1f}%</b>" for score in scores],
                textposition='outside',
                hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(text="<b>Sentiment Probability Distribution</b>", font=dict(size=15, family=FONT_FAMILY)),
        template=PLOTLY_TEMPLATE,
        height=220,
        margin=dict(l=10, r=40, t=40, b=10),
        xaxis=dict(
            range=[0, 115],
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
            ticksuffix="%",
            title=None
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=13, family=FONT_FAMILY, color="#1F2937")
        )
    )
    return fig


def create_emotion_donut_chart(emotion_data: Dict[str, Any]) -> go.Figure:
    """
    Creates an interactive Donut Chart for Emotion Distribution.
    """
    ranked = emotion_data.get('ranked_distribution', [])
    if not ranked:
        return go.Figure()

    labels = [item['emotion'] for item in ranked]
    values = [item['percentage'] for item in ranked]
    colors = [item['color'] for item in ranked]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
                textinfo='label+percent',
                textposition='inside',
                insidetextorientation='radial',
                hovertemplate="<b>%{label}</b><br>Probability: %{value:.1f}%<extra></extra>"
            )
        ]
    )

    dominant = emotion_data.get('dominant_emotion', 'Neutral')
    conf = emotion_data.get('confidence', 0.0)

    fig.update_layout(
        title=dict(text="<b>Emotion Distribution (Ekman Taxonomy)</b>", font=dict(size=15, family=FONT_FAMILY)),
        template=PLOTLY_TEMPLATE,
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        annotations=[
            dict(
                text=f"<b>{dominant}</b><br><span style='font-size:12px;color:#6B7280;'>{conf:.1f}%</span>",
                x=0.5, y=0.5,
                font=dict(size=16, family=FONT_FAMILY),
                showarrow=False
            )
        ]
    )
    return fig


def create_emotion_bar_chart(emotion_data: Dict[str, Any]) -> go.Figure:
    """
    Creates a ranked horizontal bar chart for all emotion probabilities.
    """
    ranked = emotion_data.get('ranked_distribution', [])
    if not ranked:
        return go.Figure()

    # Reversed so highest is on top
    reversed_ranked = list(reversed(ranked))
    labels = [item['emotion'] for item in reversed_ranked]
    scores = [item['percentage'] for item in reversed_ranked]
    colors = [item['color'] for item in reversed_ranked]

    fig = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=labels,
                orientation='h',
                marker=dict(color=colors, cornerradius=4),
                text=[f"<b>{s:.1f}%</b>" for s in scores],
                textposition='outside',
                hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(text="<b>Ranked Emotion Probabilities</b>", font=dict(size=15, family=FONT_FAMILY)),
        template=PLOTLY_TEMPLATE,
        height=300,
        margin=dict(l=10, r=40, t=40, b=10),
        xaxis=dict(range=[0, 115], ticksuffix="%", gridcolor='rgba(200, 200, 200, 0.2)'),
        yaxis=dict(tickfont=dict(size=12, family=FONT_FAMILY))
    )
    return fig


def create_tfidf_chart(tfidf_terms: List[Dict[str, Any]]) -> go.Figure:
    """
    Creates a horizontal bar chart displaying top TF-IDF keyword weights.
    """
    if not tfidf_terms:
        fig = go.Figure()
        fig.update_layout(
            title="<b>No distinct TF-IDF keywords found</b>",
            template=PLOTLY_TEMPLATE,
            height=200
        )
        return fig

    # Reversed for top-down display
    reversed_terms = list(reversed(tfidf_terms))
    terms = [item['term'] for item in reversed_terms]
    scores = [item['score'] for item in reversed_terms]

    fig = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=terms,
                orientation='h',
                marker=dict(
                    color=scores,
                    colorscale='Blues',
                    cornerradius=4,
                    line=dict(color='#2563EB', width=1)
                ),
                text=[f"{s:.3f}" for s in scores],
                textposition='outside',
                hovertemplate="Term: <b>%{y}</b><br>TF-IDF Weight: %{x:.4f}<extra></extra>"
            )
        ]
    )

    max_score = max(scores) if scores else 1.0
    fig.update_layout(
        title=dict(text="<b>Top TF-IDF Term Weights (Text Mining Feature Extraction)</b>", font=dict(size=15, family=FONT_FAMILY)),
        template=PLOTLY_TEMPLATE,
        height=260,
        margin=dict(l=10, r=50, t=40, b=10),
        xaxis=dict(
            range=[0, max(max_score * 1.25, 0.1)],
            gridcolor='rgba(200, 200, 200, 0.2)',
            title="TF-IDF Weight (Importance)"
        ),
        yaxis=dict(tickfont=dict(size=13, family=FONT_FAMILY))
    )
    return fig
