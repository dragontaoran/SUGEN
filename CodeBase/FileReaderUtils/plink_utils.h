// Author: paulbunn@email.unc.edu (Paul Bunn)
// Last Updated: Dec 2015
//
// Description: Class with helper functions for reading/filtering vcf files.

#include "FileReaderUtils/vcf_utils.h"

#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef PLINK_UTILS_H
#define PLINK_UTILS_H

using namespace std;

namespace file_reader_utils {

// The genotype for one (Sample, SNP).
enum GenotypeType {
  GENOTYPE_UNKNOWN,
  GENOTYPE_MISSING,
  GENOTYPE_HOMOZYGOTE_ONE,
  GENOTYPE_HOMOZYGOTE_TWO,
  // Of the following three, use the first if you don't care about the order,
  // or the 2nd two if you do.
  GENOTYPE_HETEROZYGOTE,
  GENOTYPE_HETEROZYGOTE_ONE_TWO,
  GENOTYPE_HETEROZYGOTE_TWO_ONE,
  // TODO(PHB): Consider adding more enum types here, e.g. naming them based
  // on Major/Minor Alleles (instead of generic 'one'/'two'), as well as
  // generalizing to more than bi-allelic (e.g. if up to 3 or more alleles
  // are possible for a SNP).
};

// A data structure for holding all the information about a single SNP.
struct PlinkSnpInfo {
  // Holds Position and mapping for Allele_[1 | 2] -> Nucleotide for this SNP.
  SnpInfo snp_info_;

  // The following holds the GenotypeType for each of the individuals in
  // the study for this SNP. It therefore has size = Num of Samples in Study,
  // and is indexed the same way as ordering of Samples in the Plink .fam file.
  vector<GenotypeType> genotypes_;
};

// Input used for the Read... functions.
struct PlinkInput {
  // The 3 plink files.
  string bed_filename_;
  string bim_filename_;
  string fam_filename_;

  // Specifies the set of SNPs that should be read from the plink files.
  // Exactly one of 'snps_to_read_' or ('snp_start_byte_', 'snp_start_row_')
  // should be set. In the case of the latter with the optional field
  // num_snps_to_read_ included (and larger than 1), this many *contiguous*
  // (with respect to .bed file) snps will be read. Similarly, if
  // snps_to_read_.size() == 1 and num_snps_to_read_ > 1, then
  // num_snps_to_read_ contiguous SNPs will be read.
  // Note that if using ('snp_start_byte_', 'snp_start_row_') option, each
  // one is optional, as either can be determined from the other (via functions
  // FindSnpStartByte() and FindSnpRowFromStartByte() below); if non-negative,
  // the values are used as-is (as opposed to computing them from those 2 functions).
  // When using snps_to_read_, only the SnpInfo.id_ field will be used, or if
  // it is empty, then only the SnpInfo.chromosome_ and SnpInfo.position_ fields
  // are used.
  set<SnpInfo> snps_to_read_;
  int64_t snp_start_byte_;
  int snp_start_row_;  // Row index (within .bim file) of the first SNP to read.
  // TODO(PHB): Implement the INT_MAX use-case.
  int num_snps_to_read_;  // Set to INT_MAX if you want to read rest of snps in file.

  // Holds SnpInfo (for the SNPs of interest).
  // IMPORTANT: There are exactly three valid formats that this field can have
  //            when used as input to ReadPlink():
  //              1) Empty
  //              2) In correspondence to the SNPs being read
  //              3) Full (i.e. all SNPs in .bim file)
  //            In other words, it is *not* acceptable to pass in a partially
  //            populated row_to_snp_info_ (e.g. from an earlier call to ReadPlink).
  // The indexing (order) of this vector will match the ordering of the SNPs
  // within the .bim file (when this field stores information for *all* SNPs, the
  // indexing will simply match the .bim file).
  // NOTE: Each SnpInfo within row_to_snp_info_ is expected to have 6 fields populated:
  //         id_, chromosome_, position_, row_index_, allele_one_ and allele_two_
  // NOTE: Instead of storing SnpInfo in a vector (think of this as indexing the
  //       SnpInfo items according to their Row withing .bim file), we could've
  //       stored them according to their SnpInfo sorting. However, note that both
  //       use-cases are valid, as:
  //         - Row Index -> SNP: Useful for reading in a block of contiguous SNPs
  //         - SNP -> Row Index: Useful for reading in a set of SNPs (based on id or CHR:POS)
  vector<SnpInfo> row_to_snp_info_;
  // The fields snp_pos_to_snp_info_pos_ and snp_id_to_snp_info_pos_ provide a
  // mapping between a SNP's id (resp. its CHR:POS) to its index within row_to_snp_info_.
  // NOTE: We keep maps/vectors keyed both by string (snp_id) and SnpPosition, so
  //       that user can specify either (or both) of these. Since the .bim file
  //       has both Snp ID and CHR:POS information, both fields below should
  //       be populated and complete, and it is up to the user which they use
  //       (unless a given .bim file has missing SNP ID and/or CHR:POS information).
  map<SnpPosition, int> snp_pos_to_snp_info_pos_;
  map<string, int> snp_id_to_snp_info_pos_;
  // Determines whether the entire contents of the .bim file are read, or
  // only information regarding the SNPs that are being read.
  bool read_only_required_snp_info_;
  // In case read_only_required_snp_info_ is true, the size of row_to_snp_info_
  // may not equal the total number of SNPs (which is needed to e.g. compute
  // a SNP's start byte (within .bed file) given the SNP's row in .bim file).
  // So store this quantity separately.
  // NOTE: Given a SNP's row within the .bim file, it is a quick computation
  //       to determine this SNP's start byte index (within .bed file).
  int num_snps_in_bim_file_;

  // Holds the Individual Ids, indexed in the same order is they appear
  // in the .fam file (as well as the order of PlinkSnpInfo.genotypes_).
  vector<string> samples_by_index_;

  // Whether to concatenate the first column of the .fam file (Family) when
  // constructing Individual ID, or to just use the 2nd column (Individual ID).
  bool include_family_in_id_;

  // Whether .bed file is in SNP-major or Individual-major format.
  // NOTE: Currently, only SNP-major format is supported.
  bool bed_is_snp_major_;

  PlinkInput() {
    bed_filename_ = "";
    bim_filename_ = "";
    fam_filename_ = "";
    snp_start_row_ = -1;
    snp_start_byte_ = -1;
    num_snps_to_read_ = 1;
    num_snps_in_bim_file_ = -1;
    read_only_required_snp_info_ = false;
    include_family_in_id_ = false;
    bed_is_snp_major_ = true;
  }
};

// Holds any information obtained from reading Plink Files.
struct PlinkOutput {
  // This vector has size equal to the number of SNPs requested (in the input);
  // it's order either follows the order in the input (provided a contiguous block
  // was specified), or is based on the ordering of the SnpInfo Keys (in the
  // case that the input used snps_to_read_).
  vector<PlinkSnpInfo> snp_counts_;

  // The index of the next byte to read.
  int64_t next_byte_to_read_;

  // Error Message, in case a function returned false.
  string error_msg_;

  PlinkOutput() {
    next_byte_to_read_ = -1;
    error_msg_ = "";
  }
};

class PlinkUtils {
 public:
  // Note that 'input' is not labelled as 'const', as some of its fields may
  // be updated by ReadPlink() if they are empty: samples_by_index_ and
  // snp_to_row_.
  static bool ReadPlink(PlinkInput& input, PlinkOutput* output);

 private:
  static bool ParseFamFile(
      const bool include_family_in_id, const string& fam_filename,
      vector<string>* samples_by_index, string* error_msg);

  // Parses the SNPs in the .bim file into snp_info (which is indexed according
  // to the order of the SNPs (rows) of the .bim file). In particular, each
  // SnpInfo will have 6 fields populated: id_, chromosome_, position_,
  // row_index_, allele_one_ and allele_two_; and the Value will be the row
  // index (first row has index '0') of that SNP.
  static bool ParseBimFile(
      const bool read_only_required_snp_info, const bool is_snp_major,
      const int num_samples,
      const int start_row, const int64_t start_byte, const int num_snps_to_read,
      const set<SnpInfo>& snps_to_read,
      const string& bim_filename,
      int* num_snps_in_bim_file,
      vector<SnpInfo>* row_to_snp_info,
      map<SnpPosition, int>* snp_pos_to_info_pos,
      map<string, int>* snp_id_to_info_pos,
      string* error_msg);

  // Similar to ReadPlink above, but certain fields of input (snp_to_row_ and
  // samples_by_index_) are expected to be already populated.
  static bool ParseBedFile(const PlinkInput& input, PlinkOutput* output);

  // Checks the first 3 bytes of the .bed file (first two are magic Plink version
  // identifiers, and third specifies snp-major vs. Individual major ordering),
  // and also that the file has the proper length.
  static bool CheckBedFormat(
      const bool is_snp_major, const int num_samples, const int num_snps,
      const int64_t& file_size, ifstream& bed_file, string* error_msg);

  // Determines the byte index of the input snp, and adds this to the provided map.
  // Searches snp_id_to_snp_info (resp. snp_pos_to_snp_info) to find the input
  // snp_id (resp. snp_pos); if found, populates allele_info with it.
  static bool GetSnpInfo(
      const string& snp_id, const SnpPosition& snp_pos,
      const map<string, int>& snp_id_to_snp_info_pos,
      const map<SnpPosition, int>& snp_pos_to_snp_info_pos,
      const vector<SnpInfo>& row_to_snp_info,
      SnpInfo* allele_info, string* error_msg);
 
  // Determines the byte index of the input snp, and adds this to the provided map.
  static bool GetSnpStartByteFromRowIndex(
      const bool is_snp_major, const int num_samples, const int row_index,
      int64_t* start_byte, string* error_msg);

  // Given a SNP's start byte position (within .bed file) and the number of
  // Samples, we can back compute the row-index of the SNP.
  static bool GetSnpRowIndexFromStartByte(
      const bool is_snp_major, const int num_samples, const int64_t& start_byte,
      int* row_index, string* error_msg);

  // Checks whether the given SNP matches one of the entries in 'snps_to_read'.
  static bool ShouldKeepSnpInfo(
      const string& id, const Chromosome chr, const int64_t& pos,
      const set<SnpInfo>& snps_to_read);

  // Reads one SNP's-worth of information (starting at start_byte) from the
  // (snp-major) .bed file, populating plink_snp_info with the results.
  static bool ReadPlinkSnp(
      const int num_samples, const int64_t snp_start_byte, ifstream& bed_file,
      int64_t* next_byte, PlinkSnpInfo* plink_snp_info, string* error_msg);
  // Same as above, but reads a contiguous block of 'num_snps_to_read' Snps, and
  // also (unlike ReadPlinkSnp() above) populates the PlinkSnpInfo.snp_info_ field.
  static bool ReadPlinkSnps(
      const int num_snps_to_read, const int num_samples, const int snp_start_index,
      const int64_t snp_start_byte, ifstream& bed_file,
      const vector<SnpInfo>& row_to_snp_info,
      int64_t* next_byte, vector<PlinkSnpInfo>* plink_snp_info,
      string* error_msg);


 // Unit test access to private functions.
 friend void ParseFamFileTest();
 friend void ParseBimFileTest();
};
}  // namespace file_reader_utils

#endif
