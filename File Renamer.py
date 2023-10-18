import os
folder = r'CS-IA-\H'
count = 0

for f in os.listdir(folder):
    source = os.path.join(folder, f)
    newname = os.path.join(folder, str(count) + ".jpg")
    print(newname)
    os.rename(source, newname)
    count+=1

print('All Files Renamed')

print('New Names are')
# verify the result
res = os.listdir(folder)
print(res)