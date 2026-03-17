import numpy as np


def load_file(a):
    file = np.loadtxt(a)
    type_feature= file[:, 0].astype(int)
    features = file[:, 1:]
    return type_feature, features

def normalize_features(b):
    mini = np.min(b, axis=0)
    maxi= np.max(b, axis=0)
    differ = maxi - mini
    differ[differ == 0] = 1
    return (b - mini) / differ

def Index_feature(feature_subset):
    sort_feat = sorted(feature_subset)
    a = '{'
    for i in sort_feat:
        a += str(i + 1) + ','
    a = a[:-1]
    a += '}'
    return a




def nearest_neighbor_accuracy(a, b, c):
    if len(c) == 0:
        return max(np.bincount(a))/ len(a)

    data = b[:, list(c)]
    n = len(a)
    d = 0

    for i in range(n):
        train_b = np.delete(data, i, axis=0)
        train_a = np.delete(a, i)
        test_feature = data[i]
        distances = np.linalg.norm(train_b - test_feature, axis=1)
        near= np.argmin(distances)
        goal= train_a[near]

        if goal == a[i]:
            d += 1

    return d / n


 
def forward_selection(a, b):
    feature_count = b.shape[1]
    feature_set = set()
    best_feature_set = set()
    bestacc = 0.0
 
    print("Starting forward search: \n")
    for i in range(feature_count):
        loopacc = 0.0
        best_feature = None

        for j in range(feature_count):
            if j in feature_set:
                continue
            c = feature_set | {j}
            acc = nearest_neighbor_accuracy(a, b, c)
            print("\n")
            print(f"\t Feature that we are testing: {Index_feature(c)} -- accuracy: {acc*100:.1f}%")
            if acc > loopacc:
                loopacc = acc
                best_feature = j

        feature_set.add(best_feature)

        if loopacc >= bestacc:
            bestacc = loopacc
            best_feature_set = feature_set.copy()
        else:
            print("\n(Note: Accuracy dropped. Continuing search in case of local maxima)\n")

        print(f"Best feature set at level {i+1}: {Index_feature(feature_set)}, accuracy: {loopacc*100:.1f}%\n")
    
    print(f"Finished search!! The best feature subset is {Index_feature(best_feature_set)}, which has an accuracy of {bestacc*100:.1f}%")
    return best_feature_set, bestacc
 
def backward_elimination(a, b):
    feature_count = b.shape[1]
    feature_set = set(range(feature_count))
    bestacc = nearest_neighbor_accuracy(a, b, feature_set)
    best_feature_set = feature_set.copy()
 
    print("Starting backward search: \n")
 
    for i in range(feature_count - 1):
        loopacc = 0.0
        worst_feature = None
 
        for j in list(feature_set):
            c = feature_set - {j}
            acc = nearest_neighbor_accuracy(a, b, c)
            print("\n")
            print(f"\t Feature that we are testing: {Index_feature(c)} -- accuracy: {acc*100:.1f}%")
            if acc > loopacc:
                loopacc = acc
                worst_feature = j
 
        feature_set.remove(worst_feature)
 
        if loopacc >= bestacc:
            bestacc = loopacc
            best_feature_set = feature_set.copy()
        else:
            print("\n(Note: Accuracy dropped. Continuing search in case of local maxima)\n")
 
        print(f"Best feature set at level {i+1}: {Index_feature(feature_set)}, accuracy: {loopacc*100:.1f}%\n")
 
    print(f"Search complete! Optimal feature subset: {Index_feature(best_feature_set)}, accuracy: {bestacc*100:.1f}%")
    return best_feature_set, bestacc
 
def main():
    print("Project 2: Feature Selection: ")
    filename = input("Name of the file: ").strip()

    a, b = load_file(filename)
    b = normalize_features(b)
 
    feature_count = b.shape[1]
    num_instances = b.shape[0]
 
    print(f"\nNo. of b: {feature_count} (not including the class attribute), with {num_instances} instances.")
    all_acc = nearest_neighbor_accuracy(a, b, set(range(feature_count)))
    print(f"Accuracy, using leaving-one-out evaluation, with all {feature_count} b is {all_acc*100:.1f}%\n")
 
    print("Select search algorithm:\n")
    print(" 1) Forward Selection")
    print(" 2) Backward Elimination")
    choice = input("\n").strip()
 
    if choice == '1':
        forward_selection(a, b)
    elif choice == '2':
        backward_elimination(a, b)
    else:
        print("Invalid choice.")
 
main()
