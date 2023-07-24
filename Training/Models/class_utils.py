import os, glob

path = "helmet_no_helmet_class/"

for filename in glob.glob(os.path.join(path, '*.txt')):
    with open(os.path.join(os.getcwd(), filename), 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        new_file = []
        for line in lines:
            words = line.split()
            # print(words)
            if(words[0] == '2'):
                new_line = list(line)
                new_line[0] = '1'
                line = "".join(new_line)
                new_file.append(line)
            if(words[0] == '1'):
                new_line = list(line)
                new_line[0] = '0'
                line = "".join(new_line)
                new_file.append(line) 
        f.writelines(new_file)
        f.truncate()
