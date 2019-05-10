# - Check for the presence of Etaler
#
# The following variables are set when ETALER is found:
#  HAVE_ETALER       = Set to true, if all components of ETALER
#                          have been found.
#  ETALER_INCLUDES   = Include path for the header files of ETALER
#  ETALER_LIBRARIES  = Link these to use ETALER

## -----------------------------------------------------------------------------
## Check for the header files

find_path (ETALER_INCLUDES Etaler.hpp
  PATHS /usr/local/include /usr/include /sw/include
  )

## -----------------------------------------------------------------------------
## Check for the library

find_library (ETALER_LIBRARIES Etaler
  PATHS /usr/local/lib /usr/lib /lib /sw/lib
  )

## -----------------------------------------------------------------------------
## Actions taken when all components have been found

if (ETALER_INCLUDES AND ETALER_LIBRARIES)
  set (HAVE_ETALER TRUE)
else (ETALER_INCLUDES AND ETALER_LIBRARIES)
  if (NOT ETALER_FIND_QUIETLY)
    if (NOT ETALER_INCLUDES)
      message (STATUS "Unable to find Etaler header files!")
    endif (NOT ETALER_INCLUDES)
    if (NOT ETALER_LIBRARIES)
      message (STATUS "Unable to find Etaler library files!")
    endif (NOT ETALER_LIBRARIES)
  endif (NOT ETALER_FIND_QUIETLY)
endif (ETALER_INCLUDES AND ETALER_LIBRARIES)

if (HAVE_ETALER)
  if (NOT ETALER_FIND_QUIETLY)
    message (STATUS "Found components for Etaler")
    message (STATUS "ETALER_INCLUDES = ${ETALER_INCLUDES}")
    message (STATUS "ETALER_LIBRARIES = ${ETALER_LIBRARIES}")
  endif (NOT ETALER_FIND_QUIETLY)
else (HAVE_ETALER)
  if (ETALER_FIND_REQUIRED)
    message (FATAL_ERROR "Could not find Etaler!")
  endif (ETALER_FIND_REQUIRED)
endif (HAVE_ETALER)

mark_as_advanced (
  HAVE_ETALER
  ETALER_LIBRARIES
  ETALER_INCLUDES
)