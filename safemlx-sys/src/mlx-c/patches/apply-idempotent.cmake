if(NOT DEFINED PATCH_FILE)
  message(FATAL_ERROR "PATCH_FILE must name the patch to apply")
endif()

# FetchContent may run the patch step again without restoring files changed by
# a later patch. In that case `git apply --reverse --check` cannot recognize an
# earlier patch whose context has subsequently changed, so remember successful
# applications by patch content as well.
file(SHA256 "${PATCH_FILE}" patch_hash)
get_filename_component(patch_name "${PATCH_FILE}" NAME)
set(stamp_file ".safemlx-${patch_name}.stamp")
if(EXISTS "${stamp_file}")
  file(READ "${stamp_file}" stamped_hash)
  string(STRIP "${stamped_hash}" stamped_hash)
  if(stamped_hash STREQUAL patch_hash)
    return()
  endif()
endif()

execute_process(
  COMMAND git apply --recount --reverse --check "${PATCH_FILE}"
  RESULT_VARIABLE patch_already_applied
  OUTPUT_QUIET
  ERROR_QUIET)
if(patch_already_applied EQUAL 0)
  file(WRITE "${stamp_file}" "${patch_hash}\n")
  return()
endif()

execute_process(
  COMMAND git apply --recount --check --ignore-whitespace "${PATCH_FILE}"
  RESULT_VARIABLE patch_check
  ERROR_VARIABLE patch_error)
if(NOT patch_check EQUAL 0)
  message(FATAL_ERROR "Cannot apply ${PATCH_FILE}: ${patch_error}")
endif()

execute_process(
  COMMAND git apply --recount --ignore-whitespace "${PATCH_FILE}"
  RESULT_VARIABLE patch_result
  ERROR_VARIABLE patch_error)
if(NOT patch_result EQUAL 0)
  message(FATAL_ERROR "Failed to apply ${PATCH_FILE}: ${patch_error}")
endif()

file(WRITE "${stamp_file}" "${patch_hash}\n")
