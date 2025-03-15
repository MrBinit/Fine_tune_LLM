import os
import pandas as pd
import logging

class CSVToTextConverter:
    def __init__(self, output_csv_path, txt_file_path, columns_to_select):
        """
        Initializes the CSVToTextConverter with paths to the output CSV, the output text file, and the columns to select.
        """
        self.output_csv_path = output_csv_path
        self.txt_file_path = txt_file_path
        self.columns_to_select = columns_to_select
        logging.basicConfig(level=logging.INFO)

    def load_column_info(self):
        """
        Loads the column information CSV file. If the file doesn't exist, logs an error and exits the program.

        Returns:
            pd.DataFrame: A dataframe containing column information from the CSV.
        """

        if os.path.exists(self.output_csv_path):
            return pd.read_csv(self.output_csv_path, encoding='utf-8-sig')
        else:
            logging.error(f"Error: {self.output_csv_path} not found.")
            exit()

    def check_columns_in_df(self, df):
        """
        Checks if all required columns exist in the dataframe. Logs a warning if any columns are missing.

        Args:
            df (pd.DataFrame): The dataframe to check for the required columns.

        Returns:
            list: A list of missing columns, if any.
        """

        missing_columns = [col for col in self.columns_to_select if col not in df.columns]
        if missing_columns:
            logging.warning(f"These columns are not found in the DataFrame: {', '.join(missing_columns)}")
            logging.info("Proceeding with the available columns...")
        return missing_columns

    def convert_to_text(self, csv_file_path, df_selected):
        """
        Converts the selected columns of the dataframe to text format and appends them to the output text file.

        Args:
            csv_file_path (str): The file path of the CSV file being processed.
            df_selected (pd.DataFrame): The dataframe containing only the selected columns to be converted.
        """
        with open(self.txt_file_path, 'a', encoding='utf-8') as txt_file:
            for index, row in df_selected.iterrows():
                row_str = ' '.join(map(str, row.values))
                txt_file.write(row_str + '\n')
        logging.info(f"CSV file {csv_file_path} converted to text format and appended to {self.txt_file_path}")

    def process_csv_files(self):
        """
        Processes the CSV files listed in the column information CSV. For each file, it checks for required columns,
        and converts the data into text format, appending it to the output text file.
        """

        column_info_df = self.load_column_info()

        for _, row in column_info_df.iterrows():
            csv_file_path = row['File Path']
            column_names = row['Column Names'].split(', ')
            df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
            missing_columns = self.check_columns_in_df(df)
            df_selected = df[self.columns_to_select] if all(col in df.columns for col in self.columns_to_select) else df
            self.convert_to_text(csv_file_path, df_selected)


output_csv_path = '/home/binit/fine_tune_LLama/csv_column_info.csv'
txt_file_path = '/home/binit/fine_tune_LLama/extracted_text_1.txt'
columns_to_select = ['question', 'answer', 'review', 'Article', 'Translated_Response', 'Translated_Context', 
                     'Sentences', 'correct', 'incorrect', 'article', 'ne', 'text', 'input', 'output', 
                     'Nepali Text', 'article_summary', 'Question', 'Answer', 'target']

converter = CSVToTextConverter(output_csv_path, txt_file_path, columns_to_select)
converter.process_csv_files()
