from generate_relevance_matrix import generate_relevance_matrix

if __name__ == '__main__':
    path_to_txt_file = 'story_transcription.txt'
    generate_relevance_matrix.process_events_data(path_to_txt_file)

