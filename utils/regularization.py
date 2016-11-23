def early_stop(cost_window):
    if len(cost_window) < 2:
        return False
    else:
        curr = cost_window[0]
        for idx, cost in enumerate(cost_window):
            if curr < cost or idx == 0:
                curr = cost
            else:
                return False
        return True


def early_stop2(cost_window, min_val_cost, threshold):
    if len(cost_window) < 2:
        return False
    else:
        count = 0
        for cost in cost_window:
            if cost > min_val_cost:
                count += 1
            if count == threshold:
                return True