#include <bits/stdc++.h>
using namespace std;
int main(){
    
    string s = "HiHowAreYou";
    // 
    unordered_map<char,int>mp;
    for(auto it:s)
    {
        mp[it]++;
    }
    string s1="";
    for(auto it:mp)
    {
        if(it.second==1)
            s1+=it.first;
    }
    cout<<s1<<endl;
}
