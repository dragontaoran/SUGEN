// Author: paulbunn@email.unc.edu (Paul Bunn)
// Last Updated: Dec 2015

#include "plink_utils.h"

#include "FileReaderUtils/csv_utils.h"
#include "FileReaderUtils/vcf_utils.h"
#include "MapUtils/map_utils.h"
#include "StringUtils/string_utils.h"

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace map_utils;
using namespace string_utils;
using namespace std;

namespace file_reader_utils {

bool PlinkUtils::ParseFamFile(
    const bool include_family_in_id, const string& fam_filename,
    vector<string>* samples_by_index, string* error_msg) {
  if (samples_by_index == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing .fam File: Null input.\n";
    }
    return false;
  }
  ReadCsvInput input;
  input.filename_ = fam_filename;
  input.delimiters_.insert(" ");
  input.delimiters_.insert("\\s");
  input.delimiters_.insert("\\t");
  input.has_header_ = false;
  vector<pair<int, GenericDataType>>& column_types = input.columns_to_read_;
  if (include_family_in_id) {
    column_types.push_back(make_pair(1, GenericDataType::STRING));  // Family
  }
  column_types.push_back(make_pair(2, GenericDataType::STRING));  // Individual ID
  ReadCsvOutput output;
  if (!CsvUtils::ReadCsv(input, &output)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing .fam file: Failed to parse family file '" +
                    fam_filename + "'. Error message:\n" + output.error_msg_ +
                    "\n";
    }
    return false;
  }

  const vector<vector<GenericDataHolder>>& parsed_fam_file = output.output_;
  if (parsed_fam_file.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing .fam file: Empty .fam file '" +
                    fam_filename + "'.\n";
    }
    return false;
  }

  samples_by_index->clear();
  for (int i = 0; i < parsed_fam_file.size(); ++i) {
    const vector<GenericDataHolder>& current_row = parsed_fam_file[i];
    if ((include_family_in_id && current_row.size() != 2) ||
        (!include_family_in_id && current_row.size() != 1)) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing .fam file: Unable to parse line '" +
                      Itoa(i) + ": wrong number of values found (" +
                      Itoa(static_cast<int>(current_row.size())) + ").\n";
      }
      return false;
    }
    const string& family_id = include_family_in_id ? current_row[0].str_ : "";
    const int individual_col_index = include_family_in_id ? 1 : 0;
    const string& individual_id = current_row[individual_col_index].str_;
    samples_by_index->push_back(family_id + individual_id);
  }

  return true;
}

bool PlinkUtils::ParseBimFile(
    const bool read_only_required_snp_info, const bool is_snp_major,
    const int num_samples,
    const int start_row, const int64_t start_byte, const int num_snps_to_read,
    const set<SnpInfo>& snps_to_read,
    const string& bim_filename,
    int* num_snps_in_bim_file,
    vector<SnpInfo>* row_to_snp_info,
    map<SnpPosition, int>* snp_pos_to_info_pos,
    map<string, int>* snp_id_to_info_pos,
    string* error_msg) {
  if (row_to_snp_info == nullptr || snp_pos_to_info_pos == nullptr ||
      snp_id_to_info_pos == nullptr || num_snps_in_bim_file == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing .fam file: Null input.\n";
    }
    return false;
  }

  // Only read in specific rows, if read_only_required_snp_info is true.
  int csv_start_row = -1;
  if (read_only_required_snp_info && snps_to_read.empty()) {
    if (start_row >= 0) {
      csv_start_row = start_row;
    } else if (!GetSnpRowIndexFromStartByte(
                   is_snp_major, num_samples, start_byte,
                   &csv_start_row, error_msg)) {
      return false;
    }
  }
  ReadCsvInput input;
  input.filename_ = bim_filename;
  input.delimiters_.insert(" ");
  input.delimiters_.insert("\\s");
  input.delimiters_.insert("\\t");
  input.has_header_ = false;
  if (csv_start_row >= 0) {
    input.range_to_keep_start_ = csv_start_row;
    input.range_to_keep_end_ = num_snps_to_read == INT_MAX ?
        -1 : csv_start_row + num_snps_to_read - 1;
  }
  vector<pair<int, GenericDataType>>& column_types = input.columns_to_read_;
  column_types.push_back(make_pair(1, GenericDataType::STRING));  // Chromosome
  column_types.push_back(make_pair(2, GenericDataType::STRING));  // SNP ID
  //column_types.push_back(make_pair(3, GenericDataType::INT_64));  // Distance
  column_types.push_back(make_pair(4, GenericDataType::INT_64));  // Position
  column_types.push_back(make_pair(5, GenericDataType::STRING));  // ALLELE 1
  column_types.push_back(make_pair(6, GenericDataType::STRING));  // ALLELE 2
  ReadCsvOutput output;
  if (!CsvUtils::ReadCsv(input, &output)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing .bim file: Failed to parse .bim file '" +
                    bim_filename + "'. Error message:\n" + output.error_msg_ +
                    "\n";
    }
    return false;
  }

  const vector<vector<GenericDataHolder>>& parsed_bim_file = output.output_;
  if (parsed_bim_file.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing .bim file: Empty .bim file '" +
                    bim_filename + "'.\n";
    }
    return false;
  }

  // If the entire .bim file was read, store the number of lines.
  if (csv_start_row == -1) {
    *num_snps_in_bim_file = parsed_bim_file.size();
  }
  // Some use-cases of ParseBimFile() only require num_snps_in_bim_file to be
  // set. In such cases, no need to do more work; just return.
  if (!row_to_snp_info->empty()) {
    return true;
  }

  row_to_snp_info->clear();
  snp_pos_to_info_pos->clear();
  snp_id_to_info_pos->clear();
  bool check_snp = read_only_required_snp_info && !snps_to_read.empty();
  const int num_to_keep =
      read_only_required_snp_info ? num_snps_to_read : *num_snps_in_bim_file;
  int num_kept = 0;
  for (int i = 0; i < parsed_bim_file.size(); ++i) {
    if (num_kept == num_to_keep) break;

    const vector<GenericDataHolder>& current_row = parsed_bim_file[i];
    row_to_snp_info->push_back(SnpInfo());
    SnpInfo& info = row_to_snp_info->back();
    info.row_index_ = i;
    if (!VcfUtils::ParseChromosome(current_row[0].str_, &info.chromosome_)) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing .bim file: Failed to parse Chromosome column "
                      "on line " + Itoa(i) + ": '" + current_row[0].str_ + "'.\n";
      }
      return false;
    }
    info.id_ = current_row[1].str_;
    if (info.id_.empty()) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing .bim file: Empty SNP Id "
                      "on line " + Itoa(i) + ".\n";
      }
      return false;
    }
    info.position_ = current_row[2].int64_;
    if (current_row[3].str_.length() != 1) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing .bim file: Failed to parse Allele_1 column "
                      "on line " + Itoa(i) + ": '" + current_row[3].str_ + "'.\n";
      }
      return false;
    }
    info.allele_one_ = current_row[3].str_;
    if (current_row[4].str_.length() != 1) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing .bim file: Failed to parse Allele_2 column "
                      "on line " + Itoa(i) + ": '" + current_row[4].str_ + "'.\n";
      }
      return false;
    }
    info.allele_two_ = current_row[4].str_;
    // Check if we're only supposed to read the SnpInfo for the specified SNPs,
    // and if we're in the SNPs to read were specified via set. If so, we
    // only keep the current SNP if it's in the set.
    if (check_snp) {
      if (!ShouldKeepSnpInfo(info.id_, info.chromosome_, info.position_, snps_to_read)) {
        row_to_snp_info->pop_back();
        continue;
      } else if (snps_to_read.size() == 1) {
        // check_snp was only true in the first place if snps_to_read.size() >= 1.
        // In the case that it is greater than one, we need to keep checking
        // each 'current' SNP to see if it's in the set. However, if size is
        // exactly one (and either more contiguous SNPs should be read in based on
        // num_snps_to_read, or this is the only SNP to read), then we no longer
        // need to keep checking the 'current' SNP, but will instead rely on
        // the num_kept == num_to_keep check to filter.
        check_snp = false;
      }
    }
    ++num_kept;
    SnpPosition snp_pos;
    snp_pos.chr_ = info.chromosome_;
    snp_pos.pos_ = info.position_;
    snp_pos_to_info_pos->insert(make_pair(snp_pos, i));
    snp_id_to_info_pos->insert(make_pair(info.id_, i));
  }

  return true;
}

bool PlinkUtils::ParseBedFile(const PlinkInput& input, PlinkOutput* output) {
  if (output == nullptr) return false;
  if (input.bed_filename_.empty() || input.samples_by_index_.empty() ||
      input.row_to_snp_info_.empty() ||
      input.row_to_snp_info_.size() != input.snp_id_to_snp_info_pos_.size() ||
      input.row_to_snp_info_.size() != input.snp_pos_to_snp_info_pos_.size()) {
    output->error_msg_ += "ERROR in Parsing .bed file: empty .bed file, "
                          "samples_by_index_, and/or snp_to_row_.\n";
    return false;
  }
  const int num_samples = input.samples_by_index_.size();
  const int num_snps =
      input.num_snps_in_bim_file_ > 0 ? input.num_snps_in_bim_file_ :
      input.read_only_required_snp_info_ ? -1 : input.row_to_snp_info_.size();

  // Determine which of the two accepted API formats 'input' obeys.
  const bool is_list_of_snps = !input.snps_to_read_.empty();
  const bool is_range_of_snps =
      input.snp_start_byte_ >= 0 || input.snp_start_row_ >= 0;
  if ((is_list_of_snps && is_range_of_snps) ||
      (!is_list_of_snps && !is_range_of_snps)) {
    output->error_msg_ += "ERROR in Parsing .bed file: Exactly one of "
                          "snps-to-read and snp-start-byte should be set.\n";
    return false;
  }

  // Clear output.
  output->snp_counts_.clear();

  // Open .bed file for reading.
  ifstream bed_file(input.bed_filename_, ios::in|ios::binary|ios::ate);
  if (!bed_file.is_open()) {
    output->error_msg_ += "Unable to open file '" + input.bed_filename_ + "'.\n";
    return false;
  }
  bed_file.seekg(0, ios::end);
  const int64_t file_size = bed_file.tellg();
  const int bytes_per_snp = num_samples / 4 + (3 + (num_samples % 4)) / 4;

  // Make sure .bed file has proper format (as determined by it's header).
  if (!CheckBedFormat(
          input.bed_is_snp_major_, num_samples, num_snps, file_size,
          bed_file, &output->error_msg_)) {
    return false;
  }

  // These values may be overridden inside 'is_list_of_snps' block. Additionally,
  // before they are used below, we may also intialize one of them (from the other)
  // if it is not set.
  int64_t snp_start_byte = input.snp_start_byte_;
  int snp_start_row = input.snp_start_row_;
  int num_snps_to_read =
      input.num_snps_to_read_ > 0 ? input.num_snps_to_read_ : 1;
  if (is_list_of_snps) {
    if (input.snps_to_read_.size() > 1 && input.num_snps_to_read_ > 1) {
      output->error_msg_ += "ERROR in Parsing .bed file: Unexpected input: when '"
                            "snps_to_read_' has more than one SNP, it is not "
                            "allowable to have 'num_snps_to_read_' be set.\n";
      return false;
    }
    for (const SnpInfo& snp_info : input.snps_to_read_) {
      output->snp_counts_.push_back(PlinkSnpInfo());
      PlinkSnpInfo& plink_snp_info = output->snp_counts_.back();

      SnpPosition snp_pos(snp_info.chromosome_, snp_info.position_);
      if (!GetSnpInfo(
              snp_info.id_, snp_pos,
              input.snp_id_to_snp_info_pos_, input.snp_pos_to_snp_info_pos_,
              input.row_to_snp_info_,
              &plink_snp_info.snp_info_, &output->error_msg_)) {
        return false;
      }
      snp_start_row = plink_snp_info.snp_info_.row_index_;
      if (!GetSnpStartByteFromRowIndex(
              input.bed_is_snp_major_, num_samples, snp_start_row,
              &snp_start_byte, &output->error_msg_)) {
        return false;
      }

      if (snp_start_byte + bytes_per_snp > file_size) {
        output->error_msg_ +=
            "ERROR in Parsing .bed file: File size (" + Itoa(file_size) +
            ") is not long enough to read " + Itoa(bytes_per_snp) +
            " bytes when starting at byte position " +
            Itoa(snp_start_byte) + ".\n";
        return false;
      }  

      ReadPlinkSnp(
          num_samples, snp_start_byte, bed_file,
          &(output->next_byte_to_read_), &plink_snp_info, &output->error_msg_);
    }

    // We're done if all SNPs to be read were specified in the set. Otherwise
    // (if only one SNP was specified via the set and input.num_snps_to_read_ > 1),
    // we'll continue reading the rest of the (continguous) SNPs below.
    if (input.num_snps_to_read_ <= 1) return true;
    if (num_snps_to_read != INT_MAX) num_snps_to_read--;
    snp_start_byte = output->next_byte_to_read_;
    snp_start_row++;
  }

  // Check that at least one of snp_start_byte and snp_start_row is set, and
  // set the other from it if only one is set.
  if (snp_start_byte < 0 && snp_start_row < 0) {
    output->error_msg_ += "ERROR in Parsing .bed file: Unable to get SNP info, as "
                          "neither start byte nor SNP row are available.\n";
    return false;
  }
  if (snp_start_byte < 0 &&
      !GetSnpStartByteFromRowIndex(
          input.bed_is_snp_major_, num_samples, snp_start_row,
          &snp_start_byte, &output->error_msg_)) {
    return false;
  }
  if (snp_start_row < 0 &&
      !GetSnpRowIndexFromStartByte(
          input.bed_is_snp_major_, num_samples, snp_start_byte,
          &snp_start_row, &output->error_msg_)) {
    return false;
  }

  // Handle the special case that num_snps_to_read == INT_MAX (read all
  // SNPs from the indicated start SNP through the end of the file).
  if (num_snps_to_read == INT_MAX) {
    // Check that the number of remaining bytes (from snp_start_byte)
    // is evenly divided by the number of bytes per snp.
    if ((file_size - snp_start_byte) % bytes_per_snp != 0) {
      output->error_msg_ +=
          "ERROR in Parsing .bed file: File size (" + Itoa(file_size) +
          ") minus start byte (" + Itoa(snp_start_byte) +
          ") is not evenly divided by bytes_per_snp (" +
          Itoa(bytes_per_snp) + ").\n";
      return false;
    }
    num_snps_to_read = (file_size - snp_start_byte) / bytes_per_snp;
  }
  if (snp_start_byte + (num_snps_to_read * bytes_per_snp) > file_size) {
    output->error_msg_ +=
        "ERROR in Parsing .bed file: File size (" + Itoa(file_size) +
        ") is not long enough to read " + Itoa(bytes_per_snp) +
        " bytes (for " + Itoa(num_snps_to_read) + " Samples) when starting "
        "at byte position " + Itoa(snp_start_byte) + ".\n";
    return false;
  }

  // Read in SNP genotype information.
  const int snp_start_index =
      input.read_only_required_snp_info_ ? 0 : snp_start_row;
  if (!ReadPlinkSnps(
          num_snps_to_read, num_samples, snp_start_index, snp_start_byte,
          bed_file, input.row_to_snp_info_,
          &output->next_byte_to_read_, &output->snp_counts_,
          &output->error_msg_)) {
    return false;
  }

  return true;
}

bool PlinkUtils::ShouldKeepSnpInfo(
    const string& id, const Chromosome chr, const int64_t& pos,
    const set<SnpInfo>& snps_to_read) {
  for (const SnpInfo& info : snps_to_read) {
    if (!info.id_.empty() && id == info.id_) return true;
    if ((info.id_.empty() || id.empty()) &&
        info.chromosome_ != CHROMOSOME_UNKNOWN && info.position_ >= 0 &&
        info.chromosome_ == chr && info.position_ == pos) {
      return true;
    }
  }

  return false;
}

bool PlinkUtils::CheckBedFormat(
    const bool is_snp_major, const int num_samples, const int num_snps,
    const int64_t& file_size, ifstream& bed_file, string* error_msg) {
  // First check .bed file has the expected number of bytes.
  const int bytes_per_snp = num_samples / 4 + (3 + (num_samples % 4)) / 4;
  if (num_snps >= 0 && file_size != 3 + num_snps * bytes_per_snp) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in format of .bed file: Total File Size (" +
                    Itoa(file_size) + " is not equal to 3 + num_snps (" +
                    Itoa(num_snps) + ") times bytes_per_snp (" +
                    Itoa(bytes_per_snp) + ").\n";
    }
    return false;
  }

  // Now check first 3 bytes: First two bytes should be 01101100 00011011,
  // and third byte either 00000001 (SNP-Major) or 00000000 (Individual-Major).
  bed_file.seekg(0, ios::beg);
  char* in_buffer = new char[3];
  bed_file.read(in_buffer, 3);
  const unsigned int first_byte = in_buffer[0];
  const unsigned int second_byte = in_buffer[1];
  const unsigned int third_byte = in_buffer[2];
  if (first_byte != 108 /* 108 = 01101100 */) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in format of .bed file: Bad first byte: " +
                    Itoa(first_byte) + ".\n";
    }
    return false;
  }
  if (second_byte != 27 /* 27 = 00011011 */) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in format of .bed file: Bad second byte: " +
                    Itoa(second_byte) + ".\n";
    }
    return false;
  }
  if ((is_snp_major && third_byte != 1) ||
      (!is_snp_major && third_byte != 0)) {
    if (error_msg != nullptr) {
      const string format_str = is_snp_major ? "Snp" : "Individual";
      *error_msg += "ERROR in format of .bed file: Bad third byte: " +
                    Itoa(third_byte) + " for expected " +
                    format_str + "-Major format.\n";
    }
  }

  return true;
}

bool PlinkUtils::GetSnpInfo(
    const string& snp_id, const SnpPosition& snp_pos,
    const map<string, int>& snp_id_to_snp_info_pos,
    const map<SnpPosition, int>& snp_pos_to_snp_info_pos,
    const vector<SnpInfo>& row_to_snp_info,
    SnpInfo* allele_info, string* error_msg) {
  // Sanity-check input.
  if (allele_info == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in retrieving snp info: Null input.\n";
    }
    return false;
  }

  const int* snp_info_index = nullptr;
  if (!snp_id.empty()) {
    snp_info_index = FindOrNull(snp_id, snp_id_to_snp_info_pos);
  }
  if (snp_info_index == nullptr) {
    // Lookup by id failed. Try to lookup by SnpPosition.
    snp_info_index = FindOrNull(snp_pos, snp_pos_to_snp_info_pos);
    if (snp_info_index == nullptr) {
      if (error_msg != nullptr) {
        string snp_identifier = snp_id.empty() ? "" : "ID: " + snp_id;
        string snp_pos_str =
            (snp_pos.pos_ >= 0 &&
             snp_pos.chr_ != Chromosome::CHROMOSOME_UNKNOWN) ?
            VcfUtils::PrintChromosome(snp_pos.chr_) +
            ":" + Itoa(snp_pos.pos_) : "";
        *error_msg += "ERROR in retrieving snp info: Unable to find SnpInfo for SNP: '" +
                      snp_identifier + "' " + snp_pos_str + ".\n";
      }
      return false;
    }
  }
  if (*snp_info_index < 0 || *snp_info_index >= row_to_snp_info.size()) {
    if (error_msg != nullptr) {
      string snp_identifier = snp_id.empty() ? "" : "ID: " + snp_id;
      string snp_pos_str =
          (snp_pos.pos_ >= 0 &&
           snp_pos.chr_ != Chromosome::CHROMOSOME_UNKNOWN) ?
          VcfUtils::PrintChromosome(snp_pos.chr_) +
          ":" + Itoa(snp_pos.pos_) : "";
      *error_msg += "ERROR in retrieving snp info: Unable to find SnpInfo for SNP: '" +
                    snp_identifier + "' " + snp_pos_str + "; invalid index (" +
                    Itoa(*snp_info_index) + ") for row_to_snp_info (size " +
                    Itoa(row_to_snp_info.size()) + ").\n";
    }
    return false;
  }
  *allele_info = row_to_snp_info[*snp_info_index];
  return true;
}

bool PlinkUtils::GetSnpStartByteFromRowIndex(
    const bool is_snp_major, const int num_samples, const int row_index,
    int64_t* start_byte, string* error_msg) {
  // Sanity-check input.
  if (start_byte == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in computing snp's start byte position in .bed file: "
                    "Null input.\n";
    }
    return false;
  }
  if (num_samples <= 0) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in computing snp's start byte position in .bed file: "
                    "Num Samples is zero.\n";
    }
    return false;
  }
  if (!is_snp_major) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in computing snp's start byte position in .bed file: "
                    "Non-SNP-major plink formats are not yet supported.\n";
    }
    return false;
  }

  // Each Sample's genotype takes up two bits, so you can squeeze four Samples
  // into one byte. If number Samples is not divisible by 4, the last (n % 4)
  // samples will be in the last byte, and that byte will have garbage bits to
  // complete the byte.
  const int bytes_per_snp = num_samples / 4 + (3 + (num_samples % 4)) / 4;
  // The first '3' is to skip the first three bytes of the .bed file,
  // which are header info. 
  *start_byte = 3 + row_index * bytes_per_snp;
  return true;
}

bool PlinkUtils::GetSnpRowIndexFromStartByte(
    const bool is_snp_major, const int num_samples, const int64_t& start_byte,
    int* row_index, string* error_msg) {
  if (row_index == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in GetSnpRowIndexFromStartByte: Null Input.\n";
    }
    return false;
  }

  const int64_t start_byte_w_o_header = start_byte - 3;
  const int bytes_per_snp = num_samples / 4 + (3 + (num_samples % 4)) / 4;
  if (start_byte_w_o_header < 0 ||
      start_byte_w_o_header % bytes_per_snp != 0) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in retrieving SNP tow from its start byte: Invalid start_byte "
                    "Position: position should be a multiple of the number of "
                    "bytes per SNP, which equals roof{num_samples / 4}.\n";
    }
    return false;
  }

  *row_index = start_byte_w_o_header / bytes_per_snp;

  return true;
}

bool PlinkUtils::ReadPlinkSnp(
    const int num_samples, const int64_t snp_start_byte, ifstream& bed_file, 
    int64_t* next_byte, PlinkSnpInfo* plink_snp_info, string* error_msg) {
  if (plink_snp_info == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Reading Snp at start byte " + Itoa(snp_start_byte) +
                    ": Null Input.\n";
    }
    return false;
  }

  const int bytes_to_read = num_samples / 4 + (3 + (num_samples % 4)) / 4;

  if (next_byte != nullptr) {
    *next_byte = snp_start_byte + bytes_to_read;
  }

  bed_file.seekg(snp_start_byte, ios::beg);
  char* in_buffer = new char[bytes_to_read];
  bed_file.read(in_buffer, bytes_to_read);

  int num_samples_processed = 0;
  for (int i = 0; i < bytes_to_read; ++i) {
    const unsigned int current_byte = in_buffer[i];
    for (int j = 0; j < 4; ++j) {
      const bool first_bit = (current_byte >> (j * 2)) & 1;
      const bool second_bit = (current_byte >> (j * 2)) & 2;
      const GenotypeType genotype =
          (first_bit == 0 && second_bit == 0) ? GENOTYPE_HOMOZYGOTE_ONE :
          (first_bit == 0 && second_bit == 1) ? GENOTYPE_HETEROZYGOTE :
          (first_bit == 1 && second_bit == 0) ? GENOTYPE_MISSING :
          (first_bit == 1 && second_bit == 1) ? GENOTYPE_HOMOZYGOTE_TWO :
          GENOTYPE_UNKNOWN;
      if (genotype == GENOTYPE_UNKNOWN) {
        if (error_msg != nullptr) {
          const string current_byte_str = Itoa(current_byte);
          *error_msg += "ERROR in Reading Snp at start byte " + Itoa(snp_start_byte) +
                        ": Unable to parse Genotype from "
                        "current byte: '" + current_byte_str + "'.\n";
        }
        return false;
      }
      plink_snp_info->genotypes_.push_back(genotype);
      ++num_samples_processed;
      if (num_samples_processed == num_samples) break;
    }
  }

  return true;
}

bool PlinkUtils::ReadPlinkSnps(
    const int num_snps_to_read, const int num_samples, const int snp_start_index,
    const int64_t snp_start_byte, ifstream& bed_file,
    const vector<SnpInfo>& row_to_snp_info,
    int64_t* next_byte, vector<PlinkSnpInfo>* plink_snp_info,
    string* error_msg) {
  if (plink_snp_info == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Reading plink file: Null Input.\n";
    }
    return false;
  }
  plink_snp_info->clear();

  const int bytes_per_snp = num_samples / 4 + (3 + (num_samples % 4)) / 4;
  const int bytes_to_read = bytes_per_snp * num_snps_to_read;

  if (next_byte != nullptr) {
    *next_byte = snp_start_byte + bytes_to_read;
  }

  bed_file.seekg(snp_start_byte, ios::beg);
  char* in_buffer = new char[bytes_to_read];
  bed_file.read(in_buffer, bytes_to_read);

  int num_samples_on_current_snp_processed = 0;
  int num_snps_processed = 0;
  bool start_of_next_snp = true;
  PlinkSnpInfo* current_plink_snp_info = nullptr;
  for (int i = 0; i < bytes_to_read; ++i) {
    if (start_of_next_snp) {
      plink_snp_info->push_back(PlinkSnpInfo());
      current_plink_snp_info = &plink_snp_info->back();
      current_plink_snp_info->snp_info_ =
          row_to_snp_info[snp_start_index + num_snps_processed];
      start_of_next_snp = false;
    }
    const unsigned int current_byte = in_buffer[i];
    for (int j = 0; j < 4; ++j) {
      const bool first_bit = (current_byte >> (j * 2)) & 1;
      const bool second_bit = (current_byte >> (j * 2)) & 2;
      const GenotypeType genotype =
          (first_bit == 0 && second_bit == 0) ? GENOTYPE_HOMOZYGOTE_ONE :
          (first_bit == 0 && second_bit == 1) ? GENOTYPE_HETEROZYGOTE :
          (first_bit == 1 && second_bit == 0) ? GENOTYPE_MISSING :
          (first_bit == 1 && second_bit == 1) ? GENOTYPE_HOMOZYGOTE_TWO :
          GENOTYPE_UNKNOWN;
      if (genotype == GENOTYPE_UNKNOWN) {
        if (error_msg != nullptr) {
          const string current_byte_str = Itoa(current_byte);
          *error_msg += "ERROR in Reading plink file: Unable to parse Genotype from "
                        "current byte: '" + current_byte_str + "'.\n";
        }
        return false;
      }
      current_plink_snp_info->genotypes_.push_back(genotype);
      ++num_samples_on_current_snp_processed;
      if (num_samples_on_current_snp_processed == num_samples) {
        num_samples_on_current_snp_processed = 0;
        ++num_snps_processed;
        start_of_next_snp = true;
        break;
      }
    }
  }

  return true;
}

bool PlinkUtils::ReadPlink(PlinkInput& input, PlinkOutput* output) {
  if (output == nullptr) return false;

  if ((input.num_snps_to_read_ > 1 && !input.snps_to_read_.size() > 1) ||
      (input.snps_to_read_.empty() && input.snp_start_byte_ <= 3 &&
       input.snp_start_row_ < 0)) {
    output->error_msg_ += "ERROR in Reading Plink file: Invalid input. You must either "
                          "specify all SNPs to be read in the snps_to_read_ "
                          "field, or if you want to read in a continguous block "
                          "of SNPs, you may specify just the first SNP (either "
                          "via snps_to_read_, snp_start_byte_, or snp_start_row_) "
                          "and the total number of SNPs to read.\n";
    return false;
  }

  const int num_snps_to_read =
      input.num_snps_to_read_ > 1 ? input.num_snps_to_read_ :
      input.snps_to_read_.empty() ? 1 : input.snps_to_read_.size();

  // Check that row_to_snp_info_ has one of the 3 valid input formats
  // (is empty, full, or matches the SNPs to be read).
  if (num_snps_to_read != INT_MAX &&
      input.row_to_snp_info_.size() != 0 &&
      input.row_to_snp_info_.size() != num_snps_to_read &&
      input.row_to_snp_info_.size() != input.num_snps_in_bim_file_) {
    output->error_msg_ += "ERROR in reading plink file: Invalid input. As input to "
                          "ReadPlink, field row_to_snp_info_ should either be "
                          "empty, full, or exactly match the SNPs to be read.\n";
    return false;
  }

  // Read .fam file, if necessary.
  if (input.samples_by_index_.empty()) {
    if (!ParseFamFile(
            input.include_family_in_id_, input.fam_filename_,
            &input.samples_by_index_, &output->error_msg_)) {
      return false;
    }
  }

  // Read .bim file, if necessary.
  if (input.read_only_required_snp_info_ || input.row_to_snp_info_.empty() ||
      input.num_snps_in_bim_file_ <= 0) {
    if (input.bim_filename_.empty()) {
      output->error_msg_ +=
          "ERROR in reading plink file: input's fields for storing .bim information "
          "are empty, and no .bim file was provided to populate them.\n";
      return false;
    }
    if (!ParseBimFile(
            input.read_only_required_snp_info_, input.bed_is_snp_major_,
            input.samples_by_index_.size(),
            input.snp_start_row_, input.snp_start_byte_, num_snps_to_read,
            input.snps_to_read_, input.bim_filename_,
            &input.num_snps_in_bim_file_, &input.row_to_snp_info_,
            &input.snp_pos_to_snp_info_pos_, &input.snp_id_to_snp_info_pos_,
            &output->error_msg_)) {
      return false;
    }
  }

  return ParseBedFile(input, output);
}

}  // namespace file_reader_utils
