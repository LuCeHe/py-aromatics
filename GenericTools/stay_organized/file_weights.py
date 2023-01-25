import os
from tqdm import tqdm
from humanize import naturalsize

path = input("Enter folder path: ")
ds = os.listdir(path)


# sort all folders by Mb of content and print them
def get_dir_size(path='.'):
    # print(path)
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


print(ds)
ws = []
for d in tqdm(ds):
    # print('-' * 20)
    dp = os.path.join(path, d)
    if os.path.isfile(dp):
        # print(d)
        # print(naturalsize(os.path.getsize(dp)))
        w = os.path.getsize(dp)

    else:
        w = get_dir_size(dp)

    ws.append(w)

ds = [x for _, x in sorted(zip(ws, ds))]
ws = sorted(ws)

for d, w in zip(ds, ws):
    print('-' * 20)
    print(d)
    print(naturalsize(w))

print('Total size: ', naturalsize(sum(ws)))
