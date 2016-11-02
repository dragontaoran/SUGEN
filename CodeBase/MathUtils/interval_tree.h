// Author: erik.garrison@gmail.com (Erik Garrison)
//         paulbunn@email.unc.edu (Paul Bunn)
// License: MIT
//
// Description: A class implementing the IntervalTree data structure, which
//              is optimized for storing a set of Intervals and then
//              searching over these intervals to find the subset of that are
//              intersecting/containing/contained by a given input interval.
//
// Disclaimer: This code was downloaded from https://github.com/ekg/intervaltree
//             Note that the implementation below does not optimize search
//             on the current (sub) IntervalTree: if two sorted lists were
//             maintained for each (sub) IntervalTree, then the search on each
//             (sub) IntervalTree should only take O(m + log n) time, where n
//             is the size of the (sub) IntervalTree and m is the number of 
//             matching intervals in the (sub) IntervalTree. However, the 
//             current implementation does not do this, and performs a linear
//             (O(n)) search. It wouldn't take too much work to modify the
//             code to do the faster search; but this is only really relevant
//             if there is a lot of overlap in the underlying intervals in the
//             tree. In particular, for disjoint intervals, each (sub)
//             IntervalTree has size 1, and then search time over each tree
//             is constant. Since my (current) use-case involves almost disjoint
//             intervals (the intervals on DNA corresponding to the ~10k genes),
//             I did not code the optimization (which not only would take some
//             time to implement, it also takes up more space, and is no faster
//             anyway for my use-case).

#ifndef __INTERVAL_TREE_H
#define __INTERVAL_TREE_H

#include <vector>
#include <algorithm>
#include <iostream>

namespace math_utils {

template <class T, typename K = int>
class Interval {
 public:
  K start_;
  K stop_;
  T value_;
  Interval(K s, K e, const T& v)
    : start_(s) , stop_(e), value_(v) { }
};

template <class T, typename K>
int intervalStart(const Interval<T,K>& i) {
  return i.start_;
}

template <class T, typename K>
int intervalStop(const Interval<T,K>& i) {
  return i.stop_;
}

template <class T, typename K>
std::ostream& operator<<(std::ostream& out, Interval<T,K>& i) {
  out << "Interval(" << i.start_ << ", " << i.stop_ << "): " << i.value_;
  return out;
}

template <class T, typename K = int>
class IntervalStartSorter {
 public:
  bool operator() (const Interval<T,K>& a, const Interval<T,K>& b) {
    return a.start_ < b.start_;
  }
};

template <class T, typename K = int>
class IntervalTree {
 public:
  typedef Interval<T,K> interval;
  typedef std::vector<interval> intervalVector;
  typedef IntervalTree<T,K> intervalTree;
  
  intervalVector intervals_;
  intervalTree* left_;
  intervalTree* right_;
  int center_;

  // Null constructor.
  IntervalTree<T,K>(void)
    : left_(NULL), right_(NULL), center_(0) { }

  // Copy constructor.
  IntervalTree<T,K>(const intervalTree& other) 
    : left_(NULL), right_(NULL) {
    center_ = other.center_;
    intervals_ = other.intervals_;
    if (other.left_) {
      left_ = new intervalTree(*other.left_);
    }
    if (other.right_) {
      right_ = new intervalTree(*other.right_);
    }
  }

  // Equals Comparison.
  IntervalTree<T,K>& operator=(const intervalTree& other) {
    center_ = other.center_;
    intervals_ = other.intervals_;
    if (other.left_) {
      left_ = new intervalTree(*other.left_);
    } else {
      if (left_) delete left_;
      left_ = NULL;
    }
    if (other.right_) {
      right_ = new intervalTree(*other.right_);
    } else {
      if (right_) delete right_;
      right_ = NULL;
    }
    return *this;
  }

  // Constructor via vector of intervals.
  // PHB NOTE: Input vector 'ivals' cannot be 'const' specified, as we will sort
  // it in place when constructing the IntervalTree. If it becomes necessary or
  // desired to offer a const- version, can simply have the const- version copy
  // over the input vector to a new vector, and then call this constructor on the
  // new (non-const) vector.
  IntervalTree<T,K>(
      intervalVector& ivals,
      unsigned int depth = 16,
      unsigned int minbucket = 64,
      int leftextent = 0,
      int rightextent = 0,
      unsigned int maxbucket = 512)
    : left_(NULL), right_(NULL) {
    --depth;
    IntervalStartSorter<T,K> intervalStartSorter;
    if (depth == 0 || (ivals.size() < minbucket && ivals.size() < maxbucket)) {
      std::sort(ivals.begin(), ivals.end(), intervalStartSorter);
      intervals_ = ivals;
    } else {
      if (leftextent == 0 && rightextent == 0) {
        // Sort intervals_ by start_.
        std::sort(ivals.begin(), ivals.end(), intervalStartSorter);
      }

      int leftp = 0;
      int rightp = 0;
      int centerp = 0;
      
      if (leftextent || rightextent) {
        leftp = leftextent;
        rightp = rightextent;
      } else {
        leftp = ivals.front().start_;
        std::vector<K> stops;
        stops.resize(ivals.size());
        transform(ivals.begin(), ivals.end(), stops.begin(), intervalStop<T,K>);
        rightp = *max_element(stops.begin(), stops.end());
      }

      //centerp = ( leftp + rightp ) / 2;
      centerp = ivals.at(ivals.size() / 2).start_;
      center_ = centerp;

      intervalVector lefts;
      intervalVector rights;

      for (typename intervalVector::iterator i = ivals.begin(); i != ivals.end(); ++i) {
        interval& interval = *i;
        if (interval.stop_ < center_) {
          lefts.push_back(interval);
        } else if (interval.start_ > center_) {
          rights.push_back(interval);
        } else {
          intervals_.push_back(interval);
        }
      }

      if (!lefts.empty()) {
        left_ = new intervalTree(lefts, depth, minbucket, leftp, centerp);
      }
      if (!rights.empty()) {
        right_ = new intervalTree(rights, depth, minbucket, centerp, rightp);
      }
    }
  }

  // Given interval I = (start, stop), populates 'overlapping' with the set of
  // intervals in 'this' IntervalTree that (at least partially) overlap with 'I'.
  void findOverlapping(K start, K stop, intervalVector& overlapping) const {
    if (!intervals_.empty() && !(stop < intervals_.front().start_)) {
      for (typename intervalVector::const_iterator i = intervals_.begin();
           i != intervals_.end(); ++i) {
        const interval& interval = *i;
        if (interval.stop_ >= start && interval.start_ <= stop) {
          overlapping.push_back(interval);
        }
      }
    }

    if (left_ && start <= center_) {
      left_->findOverlapping(start, stop, overlapping);
    }

    if (right_ && stop >= center_) {
      right_->findOverlapping(start, stop, overlapping);
    }
  }

  // Given interval I = (start, stop), populates 'containing' with the set of
  // intervals in 'this' IntervalTree that completely contain 'I'.
  // If start = stop, then 'I' is a point, which is a valid API call.
  void findContaining(K start, K stop, intervalVector& containing) const {
    // First check current interval tree for all relevant intervals.
    if (!intervals_.empty() && !(stop < intervals_.front().start_)) {
      for (typename intervalVector::const_iterator i = intervals_.begin();
           i != intervals_.end(); ++i) {
        const interval& interval = *i;
        if (interval.stop_ >= stop && interval.start_ <= start) {
          containing.push_back(interval);
        }
      }
    }

    if (left_ && start <= center_) {
      left_->findContaining(start, stop, containing);
    }

    if (right_ && stop >= center_) {
      right_->findContaining(start, stop, containing);
    }
  }

  // Given an interval I = (start, stop), populates 'contained' with the set
  // of intervals in 'this' IntervalTree that are contained by 'I'.
  void findContained(K start, K stop, intervalVector& contained) const {
    if (!intervals_.empty() && ! (stop < intervals_.front().start_)) {
      for (typename intervalVector::const_iterator i = intervals_.begin();
           i != intervals_.end(); ++i) {
        const interval& interval = *i;
        if (interval.start_ >= start && interval.stop_ <= stop) {
          contained.push_back(interval);
        }
      }
    }

    if (left_ && start <= center_) {
      left_->findContained(start, stop, contained);
    }

    if (right_ && stop >= center_) {
      right_->findContained(start, stop, contained);
    }
  }

  ~IntervalTree(void) {
    // Traverse the left and right trees, and delete them all the way down.
    if (left_) {
      delete left_;
    }
    if (right_) {
      delete right_;
    }
  }
};

}  // namespace math_utils

#endif
