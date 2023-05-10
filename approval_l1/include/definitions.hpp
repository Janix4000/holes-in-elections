#pragma once

#include <iostream>
#include <vector>

using namespace std;

#define all(x) (x).begin(), (x).end()

using vi = vector<int>;
using vvi = vector<vi>;
using vvvi = vector<vvi>;

using voting_hist_t = vector<int>;

void print_vec(const vector<int>& v) {
    cout << "[";
    if (v.size()) {
        cout << v.front();
    }
    for (size_t i = 1; i < v.size(); i++) {
        cout << ", " << v[i];
    }
    cout << "]";
}