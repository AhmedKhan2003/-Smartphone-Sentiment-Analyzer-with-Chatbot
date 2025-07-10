from flask import Flask, render_template, request, jsonify
import spacy
import pandas as pd
import re

app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')

new_review= pd.read_csv('new_reviews.csv')
new_phone= pd.read_csv('new_phone.csv')
database=pd.read_csv('database.csv')

def process_query(user_query):
    query_doc = nlp(user_query)
    intents = extract_intents(query_doc)

    response = ""
    for intent in intents:
        if intent == 'budget':
            budget = extract_budget(query_doc)
            relevant_phones = get_phones_within_budget(new_phone, budget)
            sorted_phones = sort_phones(relevant_phones)
            response += generate_response(sorted_phones)

        elif intent == 'brand':
            brand = extract_brand(query_doc)
            relevant_phones = get_phones_by_brand(new_phone, brand)
            sorted_phones = sort_phones(relevant_phones)
            response += generate_response(sorted_phones)

        elif intent == 'budget_between':
            budget_range = extract_budget_range(query_doc)
            if budget_range is not None:
                relevant_phones = get_phones_within_budget_range(new_phone, budget_range)
                sorted_phones = sort_phones(relevant_phones)
                response += generate_response(sorted_phones)

        elif intent == 'feature':
            feature = extract_feature(query_doc)
            relevant_phones = get_phones_with_feature(new_phone, feature)
            sorted_phones = sort_phones(relevant_phones)
            response += generate_response(sorted_phones)

        elif intent == 'statistics':
            statistics_response = generate_statistics_response(database)
            response += generate_statistics_prompt(statistics_response)

        elif intent == 'links':
            statistics_response = generate_statistics_response(database)
            response += generate_links_prompt(statistics_response)

        elif intent == 'review':
            review_sentiment = extract_review_sentiment(query_doc)
            relevant_reviews = get_reviews_by_sentiment(new_review, review_sentiment)
            sorted_reviews = sort_reviews(relevant_reviews)
            response += generate_review_response(sorted_reviews)
        
        elif intent == 'rating':
            rating = extract_float_from_string(user_query)
            relevant_phones = sort_rating(new_phone, rating)
            response += generate_rating_response(relevant_phones,rating)

        elif intent == 'greeting':
            response += generate_greeting_response()
            
        elif intent == 'phone_name':
            phone_name = extract_phone_name(query_doc)
            response += generate_response_by_phone_name(phone_name, new_phone)
            

        else:
            response += generate_general_response()

    return response

def extract_intents(query_text):
    query_doc = nlp(query_text)
    intents = []
    for token in query_doc:
        if token.lower_ in ['budget', 'price', 'affordable', 'range']:
            intents.append('budget')
        elif token.lower_ in ['brand', 'manufacturer', 'model']:
            intents.append('brand')
        elif token.lower_ in ['budget between', '-',]:
            intents.append('budget_between')
        elif token.lower_ in ['feature', 'specification', 'type']:
            intents.append('feature')
        elif token.lower_ in ['review', 'opinion', 'good', 'bad']:
            intents.append('review')
        elif token.lower_ in ['rating', 'rate', 'best']:
            intents.append('rating')
        elif token.lower_ in ['statistics', 'stats', 'data', 'numbers']:
            intents.append('statistics')
        elif token.lower_ in ['links', 'html', 'product links', 'http']:
            intents.append('links')
        elif token.lower_ in ['phone name', 'name']:
            intents.append('phone_name')
        elif token.lower_ in ['hello', 'hi', 'konichiwa']:
            intents.append('greeting')

    if not intents:
        intents.append('general')

    return intents

def extract_numbers_from_string(input_string):
    match = re.search(r'\b\d+\b', input_string)
    
    if match:
        return int(match.group())
    else:
        return None
    
def extract_float_from_string(input_string):
    match = re.search(r'\b\d+\b', input_string)
    
    if match:
        return float(match.group())
    else:
        return None

def extract_budget(query_text):
    query_text = query_text.text
    if any(keyword in query_text.lower() for keyword in ['budget', 'price', 'affordable']):
        query_doc = nlp(query_text)
        for token in query_doc:
            if token.ent_type_ == 'MONEY':
                try:
                    return float(token.text.replace('Rs', '').replace(',', ''))
                except ValueError:
                    continue
        
        numbers_in_query = extract_numbers_from_string(query_text)
        if numbers_in_query is not None:
            return float(numbers_in_query)
    
    return None


def extract_brand(query_text):
    query_text = query_text.text
    query_doc = nlp(query_text)
    for token in query_doc:
        if token.pos_ == 'PROPN':
            return token.text
    return None

def extract_feature(query_text):
    query_text = query_text.text
    query_doc = nlp(query_text)
    for token in query_doc:
        if token.lower_ in ['camera', 'battery', 'display']:
            return token.text
    return None

def extract_review_sentiment(query_text):
    query_text = query_text.text
    query_doc = nlp(query_text)
    for token in query_doc:
        if token.lower_ in ['positive', 'good', 'excellent']:
            return 'Positive'
        elif token.lower_ in ['negative', 'bad', 'poor']:
            return 'Negative'
    return 'Neutral'

def get_phones_within_budget(new_phone, budget):
    return new_phone[new_phone['Price'] <= budget]

def get_phones_by_brand(new_phone, brand):
    return new_phone[new_phone['Product Title'].str.contains(brand, case=False, na=False)]

def get_phones_with_feature(new_phone, feature):
    return new_phone[new_phone['Specifications'].str.contains(feature, case=False, na=False)]

def get_reviews_by_sentiment(new_review, sentiment):
    return new_review[new_review['Review Analysis'] == sentiment]

def sort_phones(relevant_phones):
    return relevant_phones.sort_values(by='Rating', ascending=False)

def generate_response(sorted_phones):
    if not sorted_phones.empty:
        response = "Here are some phones matching your criteria:\n"
        for _, phone in sorted_phones.iterrows():
            response += f"Title: {phone['Product Title']}, Price: {phone['Price']}, Rating: {phone['Rating']}\n"
    else:
        response = "No phones found matching your criteria."

    return response

def generate_review_response(relevant_reviews):
    if not relevant_reviews.empty:
        response = "Here are some reviews matching your criteria:\n"
        for _, review in relevant_reviews.iterrows():
            response += f"Review: {review['Reviews']}, Sentiment: {review['Review Analysis']}\n"
    else:
        response = "No reviews found matching your criteria."

    return response

def generate_greeting_response():
    response="Hello, this is a simple chatbot trained on Phones scrapped from Daraz. Please make use of it!"
    return response

def generate_general_response():
    response="I cannot answer what I do not know. Please make use of specific keywords or you'll get this message"
    return response

def generate_statistics_response(database):
    total_listings = len(database)
    average_price = database['Price'].mean()
    average_rating = database['Rating'].mean()
    database['Review_Count'] = database['Reviews'].apply(lambda x: len(x))
    average_review_count = database['Review_Count'].mean()
    top_5_highest_ratings = database.nlargest(5, 'Rating')
    top_5_most_reviews = database.nlargest(5, 'Review_Count')

    return {
        'total_listings': total_listings,
        'average_price': average_price,
        'average_rating': average_rating,
        'average_review_count': average_review_count,
        'top_5_highest_ratings': top_5_highest_ratings,
        'top_5_most_reviews': top_5_most_reviews
    }

def generate_statistics_prompt(statistics_response):
    total_listings = statistics_response['total_listings']
    average_price = statistics_response['average_price']
    average_rating = statistics_response['average_rating']
    average_review_count = statistics_response['average_review_count']
    top_5_highest_ratings = statistics_response['top_5_highest_ratings']
    top_5_most_reviews = statistics_response['top_5_most_reviews']

    response = (
        f"Statistics:\n"
        f"Total Number of Listings: {total_listings}\n"
        f"Average Product Price: {average_price:.2f}\n"
        f"Average Ratings of Products: {average_rating:.2f}\n"
        f"Average Review Count per Product: {average_review_count:.2f}\n"
        f"\nTop 5 Highest Ratings:\n{top_5_highest_ratings[['Product Title', 'Rating']].to_string(index=False)}\n"
        f"\nTop 5 Most Reviews:\n{top_5_most_reviews[['Product Title', 'Review_Count']].to_string(index=False)}"
    )

    return response

def generate_links_prompt(statistics_response):
    top_5_highest_ratings = statistics_response['top_5_highest_ratings']
    top_5_links = top_5_highest_ratings['Product Link'].tolist()

    response = "Here are the links to the top 5 highest-rated products:\n"
    for link in top_5_links:
        response += f"{link}\n"

    return response

def sort_rating(relevant_phones, rating):
    relevant_phones = relevant_phones[relevant_phones['Rating'] >= rating]
    sorted_phones = relevant_phones.sort_values(by='Rating', ascending=False)
    return sorted_phones

def generate_rating_response(sorted_phones):
    columns_to_display = ['Product Title', 'Price', 'Rating']
    
    if not sorted_phones.empty:
        response = f"Here are some phones with a rating greater than or equal to {rating}:\n"
        for _, phone in sorted_phones.iterrows():
            response += f"Title: {phone['Product Title']}, Price: {phone['Price']}, Rating: {phone['Rating']}\n"
        response += f"\nColumns Displayed: {', '.join(columns_to_display)}"
    else:
        response = f"No phones found with a rating greater than or equal to {rating}."

    return response

def generate_response_by_phone_name(phone_name, new_phone):
    matching_phones = new_phone[new_phone['Product Title'].str.contains(phone_name, case=False, na=False)]

    if not matching_phones.empty:
        response = f"Here are the phones with the name '{phone_name}':\n"
        for _, phone in matching_phones.iterrows():
            response += f"Title: {phone['Product Title']}, Price: {phone['Price']}, Rating: {phone['Rating']}\n"
    else:
        response = f"No phones found with the name '{phone_name}'."

    return response

def extract_phone_name(query_text):
    query_doc = nlp(query_text)
    for ent in query_doc.ents:
        if ent.label_ == 'PRODUCT':
            return ent.text
    for token in query_doc:
        if token.pos_ == 'PROPN':
            return token.text
    return None

def extract_budget_range(query_text):
    query_text = query_text.text
    query_doc = nlp(query_text)
    tokens = [token.text.lower() for token in query_doc]

    if 'between' in tokens and 'and' in tokens:
        index_between = tokens.index('between')
        index_and = tokens.index('and')

        if index_between < index_and - 1 and index_and < len(tokens) - 1:
            lower_bound = extract_numbers_from_string(tokens[index_between + 1])
            upper_bound = extract_numbers_from_string(tokens[index_and + 1])
            
            if lower_bound is not None and upper_bound is not None:
                return lower_bound, upper_bound

    return None

def get_phones_within_budget_range(new_phone, budget_range):
    lower_bound, upper_bound = budget_range
    return new_phone[(new_phone['Price'] >= lower_bound) & (new_phone['Price'] <= upper_bound)]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_bot_response():
    user_text = request.args.get('msg')
    response = process_query(user_text)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
