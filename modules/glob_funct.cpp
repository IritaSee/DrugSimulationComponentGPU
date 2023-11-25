#include "glob_funct.hpp"
// #include "../libs/zip.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// to make it more "portable" between OSes.
#if defined _WIN32
  #include <direct.h>
  #define snprintf _snprintf
  #define vsnprintf _vsnprintf
  #define strcasecmp _stricmp
  #define strncasecmp _strnicmp
#else
  #include <dirent.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


void mpi_printf(unsigned short node_id, const char *fmt, ...)
{

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);	
}

void mpi_fprintf(unsigned short node_id, FILE *stream, const char *fmt, ...)
{
#ifndef _WIN32
  if(mympi::rank == node_id){
    va_list args;
    va_start(args, fmt);
    vfprintf(stream, fmt, args);
    va_end(args);
  }
#else
  va_list args;
  va_start(args, fmt);
  vfprintf(stream, fmt, args);
  va_end(args);	
#endif
}

void edison_assign_params(int argc, char *argv[], param_t *p_param)
{
  bool is_default;
  char buffer[100];
  char key[100];
  char value[100];
  char file_name[150];
  FILE *fp_inputdeck;

  // parameters from arguments
  for (int idx = 1; idx < argc; idx += 2) {
    if (!strcasecmp(argv[idx], "-input_deck"))
      strncpy(file_name, argv[idx + 1], sizeof(file_name));
    else if (!strcasecmp(argv[idx], "-hill_file"))
      strncpy(p_param->hill_file, argv[idx + 1], sizeof(p_param->hill_file));
    // else if (!strcasecmp(argv[idx], "-cvar_file"))
    //   strncpy(p_param->hill_file, argv[idx + 2], sizeof(p_param->cvar_file));
  }  

  is_default = false;
  fp_inputdeck = fopen( file_name, "r");
  if(fp_inputdeck == NULL){
    fprintf(stderr, "Cannot open input deck file %s!!!\nUse default value as the failsafe.\n", file_name);
    is_default = true;
  }

  // read input_deck line by line
  // and store each line to the buffer
  while ( is_default == false && fgets( buffer, 100, fp_inputdeck ) != NULL ) {
    sscanf( buffer, "%s %*s %s", key, value );
    if (strcasecmp(key, "Simulation_Mode") == 0) {
      p_param->simulation_mode = strtod( value, NULL );
    }
    else if (strcasecmp(key, "Celltype") == 0) {
      p_param->celltype = strtod( value, NULL );
    }
    else if (strcasecmp(key, "Is_Dutta") == 0) {
      p_param->is_dutta = strtol( value, NULL, 10 );
    }
    else if (strcasecmp(key, "Use_Conductance_Variability") == 0) {
      p_param->is_cvar = strtol( value, NULL, 10 );
    }
    else if (strcasecmp(key, "Pace_Find_Steepest") == 0) {
      p_param->find_steepest_start = strtod( value, NULL);
    }
    else if (strcasecmp(key, "GPU_Index") == 0) {
      p_param->gpu_index = strtod( value, NULL);
    }
    else if (strcasecmp(key, "Basic_Cycle_Length") == 0) {
      p_param->bcl = strtod( value, NULL );
    }
    else if (strcasecmp(key, "Number_of_Pacing") == 0) {
      p_param->pace_max = strtod( value, NULL );
    }
    else if (strcasecmp(key, "Time_Step") == 0) {
      p_param->dt = strtod( value, NULL );
    }
    //TODO: #Automation 1. eliminate drug_name and concentration on drug_name
    // else if (strcasecmp(key, "Drug_Name") == 0) {
    //   strncpy( p_param->drug_name, value, sizeof(p_param->concs) );
    // }
    // else if (strcasecmp(key, "Concentrations") == 0) {
    //   p_param->conc = strtod( value, NULL );
    // }

  }

  if( is_default == false ) fclose( fp_inputdeck );
}

// #ifndef _WIN32
// int extract_zip(const char *zip_filename, const char *target_folder)
// {
//   // return zip_extract( zip_filename, target_folder, NULL, NULL);
// }

// void create_zip(const char *zip_filename, const char *target_folder)
// {
//   // struct zip_t *zp_result = zip_open(zip_filename, ZIP_DEFAULT_COMPRESSION_LEVEL, 'w');
//   // if( zp_result != NULL ){
//   //   zip_walk(zp_result, target_folder);
//   // }
//   // else{
//   //   fprintf(stderr, "Error when zipping result folder!!\n");
//   //   return;
//   // }
//   // if( zp_result != NULL ) zip_close(zp_result);
// }

// void zip_walk(struct zip_t *zip, const char *path) 
// {
//   // DIR *dir;
//   // struct dirent *entry;
//   // char fullpath[MAX_PATH];
//   // struct stat s;

//   // memset(fullpath, 0, MAX_PATH);
//   // dir = opendir(path);
//   // if(dir == NULL){
//   //   fprintf(stderr, "Error opening the directory %s\n", path);
//   //   return;
//   // }


//   // while ((entry = readdir(dir))) {
//   //   // skip "." and ".."
//   //   if (!strcasecmp(entry->d_name, ".\0") || !strcasecmp(entry->d_name, "..\0"))
//   //     continue;

//   //   snprintf(fullpath, sizeof(fullpath), "%s/%s", path, entry->d_name);
//   //   stat(fullpath, &s);
//   //   if (S_ISDIR(s.st_mode))
//   //     zip_walk(zip, fullpath);
//   //   else {
//   //     zip_entry_open(zip, fullpath);
//   //     zip_entry_fwrite(zip, fullpath);
//   //     zip_entry_close(zip);
//   //   }
//   // }
//   // closedir(dir);
// }

// void remove_dir_content(const char *path)
// {
  // struct dirent *de;
  // char fname[300];
  // DIR *dr = opendir(path);
  // if(dr == NULL){
  //   printf("No file or directory found\n");
  //   return;
  // }
  // while((de = readdir(dr)) != NULL){
  //   int ret = -1;
  //   struct stat statbuf;
  //   sprintf(fname,"%s/%s",path,de->d_name);
  //   if(!strcasecmp(de->d_name, ".") || !strcasecmp(de->d_name, "..")) continue;
  //   if(!stat(fname, &statbuf)){
  //     if(S_ISDIR(statbuf.st_mode)){
  //       //printf("Is dir: %s\n",fname);
  //       if(ret != 0){
  //         remove_dir_content(fname);
  //         rmdir(fname);
  //       }
  //     }
  //     else{
  //       unlink(fname);
  //       //printf("Is file: %s\n",fname);
  //       //printf("Err: %d\n",unlink(fname));
  //     }
  //   }
  // }
  // closedir(dr);
// }
// #endif

int make_directory(const char* dirname )
{
#if defined _WIN32
  return _mkdir(dirname);
#else
  return mkdir(dirname, 0775);
#endif	
}

int is_file_existed(const char* pathname)
{
#if defined _WIN32
  struct _stat buf;
  return _stat( pathname, &buf );
#else
  struct stat st = {0};
  return stat(pathname, &st);
#endif
}
