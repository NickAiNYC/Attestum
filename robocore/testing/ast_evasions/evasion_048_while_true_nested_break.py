# @reject_reason: unbounded loop
while True:
    for i in range(3):
        if i == 2:
            break
