// Date: May 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description: Helper functions for Maps.

#ifndef MAP_UTILS_H
#define MAP_UTILS_H

#include <cstdlib>
#include <map>
#include <set>

using namespace std;

namespace map_utils {

// Returns the set of Keys for the input map.
template<typename Key, typename Value>
inline set<Key> Keys(const map<Key, Value>& input) {
  set<Key> to_return;
  for (const pair<Key, Value>& itr : input) {
    to_return.insert(itr.first);
  }
  return to_return;
}

// Inserts ('key', 'value') to the indicated map, overwriting any Value that
// may already be present with that 'key'.
template<typename Key, typename Value>
inline void InsertOrReplace(map<Key, Value>& input, Key k, Value v) {
  pair<typename map<Key, Value>::iterator, bool> insert_info =
      input.insert(make_pair(k, v));
  if (!insert_info.second) insert_info.first->second = v;
}

// Either returns a pointer to the existing Value corresponding to 'key' (after
// inserting ('key', 'value') into the map if 'key' was not already present.
template<typename Key, typename Value>
inline Value* FindOrInsert(
    const Key& key, map<Key, Value>& input, const Value& default_value) {
  typename map<Key, Value>::iterator itr = input.find(key);
  if (itr == input.end()) {
    return &(input.insert(make_pair(key, default_value)).first->second);
  }
  return &(itr->second);
}

// Same as above, but doesn't modify the map (if key does not exist, just
// returns the default value).
template<typename Key, typename Value>
inline Value FindWithDefault(
    const Key& key, const map<Key, Value>& input, const Value& default_value) {
  typename map<Key, Value>::const_iterator itr = input.find(key);
  if (itr == input.end()) {
    return default_value;
  }
  return itr->second;
}

// Returns a pointer to the Value corresponding to 'key', or nullptr
// if 'key' does not exist in the input map.
template<typename Key, typename Value>
inline Value* FindOrNull(const Key& key, map<Key, Value>& input) {
  typename map<Key, Value>::iterator itr = input.find(key);
  if (itr == input.end()) return nullptr;
  return &(itr->second);
}
// Same above, but returns const reference.
template<typename Key, typename Value>
inline const Value* FindOrNull(const Key& key, const map<Key, Value>& input) {
  typename map<Key, Value>::const_iterator itr = input.find(key);
  if (itr == input.end()) return nullptr;
  return &(itr->second);
}

}  // namespace map_utils

#endif
