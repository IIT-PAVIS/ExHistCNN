import csv


def read_csv(csv_file_path):
    csv_data = []
    with open(csv_file_path, 'r') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # Column names
                line_count += 1
            else:
                csv_data.append(row)
                line_count += 1
    csvFile.close()

    return csv_data


def save_csv(csv_file_path,lines):
    with open(csv_file_path, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(lines)
    csvFile.close()