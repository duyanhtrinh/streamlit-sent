import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load data
df = pd.read_csv('ShopeeFood_cleaned_predicted.csv')

# Define a function to generate wordcloud Negative words for selected restaurant id in
def generate_wordcloud_negative(restaurant_id):
    negative_words = df[(df['IDRestaurant'] == restaurant_id) & (df['Predicted Sentiment'] == 'Negative')]['Comment_new'].str.cat(sep=' ')
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(negative_words)
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(fig)

# Define a function to generate wordcloud Positive words for selected restaurant id
def generate_wordcloud_positive(restaurant_id):
    negative_words = df[(df['IDRestaurant'] == restaurant_id) & (df['Predicted Sentiment'] == 'Positive')]['Comment_new'].str.cat(sep=' ')
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(negative_words)
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(fig)

# define a function to count top 10 negative and positive words for a restaurant in a dataframe
def top_words_restaurant(IDRestaurant):
    # take only comments with IDRestaurant and Label_pred negative
    df_restaurant_negative = df[(df['IDRestaurant'] == IDRestaurant) & (df['Predicted Sentiment'] == 'Negative')]

    # get the top 10 negative words
    top_negative_words = df_restaurant_negative['Comment_new'].str.split(expand=True).stack().value_counts().head(10).reset_index()
    top_negative_words.columns = ['Negative Words', 'Count']

    # take only comments with IDRestaurant and Label_pred positive
    df_restaurant_positive = df[(df['IDRestaurant'] == IDRestaurant) & (df['Predicted Sentiment'] == 'Positive')]

    # get the top 10 positive words
    top_positive_words = df_restaurant_positive['Comment_new'].str.split(expand=True).stack().value_counts().head(10).reset_index()
    top_positive_words.columns = ['Positive Words', 'Count']

    return top_negative_words, top_positive_words

# Title
st.title("Shopee Food Sentiment Analysis")
menu = ["Introduction page", "Guide page","Analytics Dashboard"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Introduction page':    
    st.subheader("Giới thiệu GUI Sentiment Analysis")
    st.write("---")
    st.subheader("Mục tiêu:")
    st.write("Xây dựng hệ thống hỗ trợ nhà hàng/quán ăn phân loại các phản hồi của khách hàng thành các nhóm: tích cực, tiêu cực, trung tính dựa trên dữ liệu dạng văn bản.")
    st.write("Giúp người dùng có cái nhìn tổng quan về quán ăn để có thể đưa ra những lựa chọn phù hợp cho mình.")
    st.write("---")
elif choice == "Guide page":
    st.subheader("Hướng dẫn điều khiển")
    st.write("Chọn ID của nhà hàng muốn analyze (Chọn thông qua dropdown menu): ")
    col1, col2 = st.columns(2)
    restaurant_id = col1.selectbox('Select Restaurant ID:', df['IDRestaurant'].unique())
    st.write("Nhấp vào từng dropdown bar để xem thông tin:")
    st.write("""
    1. Thông tin nhà hàng
    2. Sự phân bố của rating
    3. Thống kê phân loại cảm xúc khách hàng
    4. Thống kê thời gian khách hàng đưa ra rating
        """)
    st.write("---")
    st.write("Tick vào từng box để hiển thị thêm thông tin.")
    st.write("""
    1. Vẽ WordCloud những từ xuất hiện nhiều nhất
    2. Hiển thị những từ xuất hiện nhiều nhất trong nhóm positive và negative.
    3. Hiển thị những comment của nhà hàng đó
   
        """)
elif choice == 'Analytics Dashboard':

    st.subheader("Analytics Dashboard")

    # Allow user to select restaurant ID:
    col1, col2 = st.columns(2)
    restaurant_id = col1.selectbox('Select Restaurant ID:', df['IDRestaurant'].unique())
    col2.write("")
    col2.write("")
    col2.write(f"Selected Restaurant ID: {restaurant_id}")

    # Display the name, address, opening hours and the average rating of the selected restaurant in an expandable container
    with st.expander("Restaurant Information"):
        restaurant_name = df[df['IDRestaurant'] == restaurant_id]['Restaurant'].values[0]
        restaurant_address = df[df['IDRestaurant'] == restaurant_id]['Address'].values[0]
        restaurant_avg_rating = df[df['IDRestaurant'] == restaurant_id]['Rating'].mean()
        st.write(f"Restaurant Name: {restaurant_name}")
        st.write(f"Restaurant Address: {restaurant_address}")
        st.write(f"Average Rating: {restaurant_avg_rating:.1f} / 10")

    # Display the count of each rating for the selected restaurant in a progress bar
    with st.expander("Rating Distribution"):
        rating_counts = df[df['IDRestaurant'] == restaurant_id]['Rating'].apply(lambda x: round(x + 0.5) if x > 0.5 else round(x)).value_counts().sort_index()
        for i in range(1, 11):
            count = rating_counts[i] if i in rating_counts else 0
            st.write(f"Rating {i} ({count} reviews)")
            st.progress(count / rating_counts.sum())

    # Display the number of positive and negative comments for the selected restaurant in an expandable container
    with st.expander("Sentiment Analysis"):
        num_comments = df[df['IDRestaurant'] == restaurant_id].shape[0]
        num_positive_comments = df[(df['IDRestaurant'] == restaurant_id) & (df['Predicted Sentiment'] == 'Positive')].shape[0]
        num_negative_comments = df[(df['IDRestaurant'] == restaurant_id) & (df['Predicted Sentiment'] == 'Negative')].shape[0]
        st.write(f"Number of Comments: {num_comments}")
        st.write(f"Number of Positive Comments: {num_positive_comments}")
        st.write(f"Number of Negative Comments: {num_negative_comments}")

    # Show how likely the comments are positive or negative on PM or AM based on column 'Time'
    with st.expander("Time Analysis"):
        df['Time'] = pd.to_datetime(df['Time'])
        df['AM/PM'] = df['Time'].dt.strftime('%p')
        am_pm = df[df['IDRestaurant'] == restaurant_id]['AM/PM'].value_counts()
        st.write("### Number of Comments in AM and PM")
        st.write(am_pm)
        st.write("### Number of Positive and Negative Comments in AM and PM")
        positive_comments_am = df[(df['IDRestaurant'] == restaurant_id) & (df['Predicted Sentiment'] == 'Positive') & (df['AM/PM'] == 'AM')].shape[0]
        positive_comments_pm = df[(df['IDRestaurant'] == restaurant_id) & (df['Predicted Sentiment'] == 'Positive') & (df['AM/PM'] == 'PM')].shape[0]
        negative_comments_am = df[(df['IDRestaurant'] == restaurant_id) & (df['Predicted Sentiment'] == 'Negative') & (df['AM/PM'] == 'AM')].shape[0]
        negative_comments_pm = df[(df['IDRestaurant'] == restaurant_id) & (df['Predicted Sentiment'] == 'Negative') & (df['AM/PM'] == 'PM')].shape[0]
        # Create a dataframe to store the number of positive and negative comments in AM and PM
        data = {
            'Time': ['AM', 'PM'],
            'Positive Comments': [positive_comments_am, positive_comments_pm],
            'Negative Comments': [negative_comments_am, negative_comments_pm]
        }
        df_comments = pd.DataFrame(data, index=['AM', 'PM'])
        st.dataframe(df_comments, hide_index=True)

    # Display the Wordcloud for Negative and Positive words
    if st.checkbox('Generate Wordcloud'):
        st.write("### Wordcloud for Negative Words")
        generate_wordcloud_negative(restaurant_id)
        st.write("### Wordcloud for Positive Words")
        generate_wordcloud_positive(restaurant_id)

    if st.checkbox('Show Top Words'):
        top_negative_words, top_positive_words = top_words_restaurant(restaurant_id)
        col1, col2 = st.columns(2)
        col1.write("### Top 10 Negative Words")
        col1.write(top_negative_words)
        col2.write("### Top 10 Positive Words")
        col2.write(top_positive_words)

    # show all comments for the selected restaurant
    if st.checkbox('Show Comments'):
        st.write("### Comments")
        comments = df[df['IDRestaurant'] == restaurant_id]['Comment']
        st.write(comments)
    