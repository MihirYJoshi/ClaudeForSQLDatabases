import pymysql
import pandas as pd
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# Database connection parameters
connection_params = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'Scouting24',
    'port': 8889
}

# Initialize Anthropic client
anthropic = Anthropic(api_key="")

def generate_sql_query(prompt, conversation_history):
    response = anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system = "You are an SQL query generator. Use the provided schema and prompt to generate accurate and relevant SQL queries that will get all the necessary information to answer the prompt.",
        messages=[
            {
                "role": "user", 
                "content": [
                    {
                        "type":"text",
                        "text": f"""Given the following schema for `24_3476_caoc_data` table: 
                            [`matchKey` varchar(60) NOT NULL PRIMARY KEY (teamNumber-matchNumber), `scout` varchar(60) NOT NULL (scouter name), 
                            `matchNumber` varchar(10) NOT NULL (Match Number), `teamNumber` varchar(10) NOT NULL (Team Number), 
                            `autoMobility` tinyint(1) NOT NULL (0: Did not Exit Zone During Autonomous Mode, 1:Exited Zone During Autonomous Mode), `autoAmpNote` smallint(6) NOT NULL (Scored in AMP During Autonomous Mode), 
                            `autoSpeakerNote` smallint(6) NOT NULL (Scored in Speaker During Autonomous Mode), `autoPath` longtext NOT NULL (Empty Column), 
                            `teleopAmpNote` smallint(6) NOT NULL (Scored in Amp During Teleoperated Mode), `teleopSpeaker` smallint(6) NOT NULL (Scored in Speaker During Teleoperated Mode), 
                            `teleopSpeakerAmplified` smallint(6) NOT NULL (Scored in Speaker During Teleoperated Mode While the Speaker was Amplified), 
                            `teleopTrap` smallint(6) NOT NULL (Scored in Trap During Teleoperated Mode), `climb` varchar(100) DEFAULT NULL (NONE: No climb, PARKED: No climb but in zone, ONSTAGE: Climbed on Chain), 
                            `climbSpotlighted` tinyint(1) NOT NULL (0: Not Spotlighted, 1: Climbed on Chain and Spotlighted), `climbHarmony` tinyint(1) NOT NULL (0: Not Climbed on Chain With Others, 1: Climbed on Chain With Others), 
                            `cannedComments` text DEFAULT NULL (Comments in Multiple Choice Format), `textComments` text DEFAULT NULL (Comments in Free Response Format)];

                            Conversation history:
                            {conversation_history}

                            Generate an SQL query that will get all the information necessary to answer the following prompt: {prompt}
                            
                            Provide only the SQL query without any additional explanation."""
                    }
                ]
            }
        ]
    )
    return response.content[0].text.strip()


def generate_final_answer(prompt, query_result, conversation_history):
    response = anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are an assistant that interprets SQL query results and provides answers to user prompts. Refer to previous context when appropriate.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":
                            f"""Prompt: {prompt};
                            Conversation history: {conversation_history};

                            Table Schema:
                            [`matchKey` varchar(60) NOT NULL PRIMARY KEY (teamNumber-matchNumber), `scout` varchar(60) NOT NULL (scouter name), 
                            `matchNumber` varchar(10) NOT NULL (Match Number), `teamNumber` varchar(10) NOT NULL (Team Number), 
                            `autoMobility` tinyint(1) NOT NULL (0: Did not Exit Zone During Autonomous Mode, 1:Exited Zone During Autonomous Mode), `autoAmpNote` smallint(6) NOT NULL (Scored in AMP During Autonomous Mode), 
                            `autoSpeakerNote` smallint(6) NOT NULL (Scored in Speaker During Autonomous Mode), `autoPath` longtext NOT NULL (Empty Column), 
                            `teleopAmpNote` smallint(6) NOT NULL (Scored in Amp During Teleoperated Mode), `teleopSpeaker` smallint(6) NOT NULL (Scored in Speaker During Teleoperated Mode), 
                            `teleopSpeakerAmplified` smallint(6) NOT NULL (Scored in Speaker During Teleoperated Mode While the Speaker was Amplified), 
                            `teleopTrap` smallint(6) NOT NULL (Scored in Trap During Teleoperated Mode), `climb` varchar(100) DEFAULT NULL (NONE: No climb, PARKED: No climb but in zone, ONSTAGE: Climbed on Chain), 
                            `climbSpotlighted` tinyint(1) NOT NULL (0: Not Spotlighted, 1: Climbed on Chain and Spotlighted), `climbHarmony` tinyint(1) NOT NULL (0: Not Climbed on Chain With Others, 1: Climbed on Chain With Others), 
                            `cannedComments` text DEFAULT NULL (Comments in Multiple Choice Format), `textComments` text DEFAULT NULL (Comments in Free Response Format)];

                            SQL query result:
                            {query_result};
                            Based the above information, provide an answer with an in-depth explanation."""
                    }
                ]
            }
        ]
    )
    return response.content[0].text.strip()
        

def execute_query(sql):
    try:
        connection = pymysql.connect(**connection_params)
        with connection.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
        return pd.DataFrame(results, columns=columns)
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return None

def main():
    conversation_history = []
    while True:
        prompt = input("Enter your query about the 24_3476_caoc_data table (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break

        conversation_history.append(f"Human: {prompt}")

        # Generate SQL query
        sql_query = generate_sql_query(prompt, "\n".join(conversation_history))
        print(sql_query)
        
        # Execute the query
        query_result = execute_query(sql_query)
        if query_result is not None:
            print("\nQuery Result:")
            print(query_result)
            
            # Generate final answer
            final_answer = generate_final_answer(prompt, query_result.to_string(), "\n".join(conversation_history))
            print("\nClaude: " + final_answer)
            conversation_history.append(f"Claude: {final_answer}")
        else:
            print("Claude: I'm sorry, but I couldn't execute that query. Could you please rephrase your question?")
            conversation_history.append("Claude: I'm sorry, but I couldn't execute that query. Could you please rephrase your question?")
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()