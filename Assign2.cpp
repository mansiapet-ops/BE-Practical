#include <iostream>
#include <omp.h>
#include <cstdlib>
using namespace std;

#define N 10000

// -------- Sequential Bubble Sort --------
void bubbleSortSequential(int arr[], int n) {
    for(int i = 0; i < n-1; i++) {
        for(int j = 0; j < n-i-1; j++) {
            if(arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

// -------- Parallel Bubble Sort --------
void bubbleSortParallel(int arr[], int n) {
    for(int i = 0; i < n; i++) {
        if(i % 2 == 0) {
            #pragma omp parallel for
            for(int j = 0; j < n-1; j += 2) {
                if(arr[j] > arr[j+1])
                    swap(arr[j], arr[j+1]);
            }
        } else {
            #pragma omp parallel for
            for(int j = 1; j < n-1; j += 2) {
                if(arr[j] > arr[j+1])
                    swap(arr[j], arr[j+1]);
            }
        }
    }
}

// -------- Merge Function --------
void merge(int arr[], int low, int mid, int high) {
    int *temp = new int[high-low+1];
    int i = low, j = mid+1, k = 0;

    while(i <= mid && j <= high) {
        if(arr[i] < arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }

    while(i <= mid) temp[k++] = arr[i++];
    while(j <= high) temp[k++] = arr[j++];

    for(i = low, k = 0; i <= high; i++, k++)
        arr[i] = temp[k];

    delete[] temp;
}

// -------- Sequential Merge Sort --------
void mergeSortSequential(int arr[], int low, int high) {
    if(low < high) {
        int mid = (low + high)/2;
        mergeSortSequential(arr, low, mid);
        mergeSortSequential(arr, mid+1, high);
        merge(arr, low, mid, high);
    }
}

// -------- Parallel Merge Sort --------
void mergeSortParallel(int arr[], int low, int high, int depth = 0) {
    if(low < high) {
        int mid = (low + high)/2;

        if(depth < 4) { // LIMIT THREAD CREATION
            #pragma omp parallel sections
            {
                #pragma omp section
                mergeSortParallel(arr, low, mid, depth+1);

                #pragma omp section
                mergeSortParallel(arr, mid+1, high, depth+1);
            }
        } else {
            mergeSortSequential(arr, low, mid);
            mergeSortSequential(arr, mid+1, high);
        }

        merge(arr, low, mid, high);
    }
}

// -------- Main --------
int main() {
    int *arr1 = new int[N];
    int *arr2 = new int[N];
    int *arr3 = new int[N];
    int *arr4 = new int[N];

    // Initialize array
    for(int i = 0; i < N; i++) {
        int val = rand() % 10000;
        arr1[i] = arr2[i] = arr3[i] = arr4[i] = val;
    }

    double start, end;

    // Sequential Bubble
    start = omp_get_wtime();
    bubbleSortSequential(arr1, N);
    end = omp_get_wtime();
    double seqBubble = end - start;

    // Parallel Bubble
    start = omp_get_wtime();
    bubbleSortParallel(arr2, N);
    end = omp_get_wtime();
    double parBubble = end - start;

    // Sequential Merge
    start = omp_get_wtime();
    mergeSortSequential(arr3, 0, N-1);
    end = omp_get_wtime();
    double seqMerge = end - start;

    // Parallel Merge
    start = omp_get_wtime();
    mergeSortParallel(arr4, 0, N-1);
    end = omp_get_wtime();
    double parMerge = end - start;

    // Output
    cout << "\nSequential Bubble: " << seqBubble;
    cout << "\nParallel Bubble: " << parBubble;
    cout << "\nBubble Speedup: " << seqBubble/parBubble << endl;

    cout << "\nSequential Merge: " << seqMerge;
    cout << "\nParallel Merge: " << parMerge;
    cout << "\nMerge Speedup: " << seqMerge/parMerge << endl;

    delete[] arr1;
    delete[] arr2;
    delete[] arr3;
    delete[] arr4;

    return 0;
}