#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include<string>
int is_file_exist(std::string file_name);
int is_file_exist(std::string file_name)
{
    FILE* fp = fopen(file_name.c_str(), "r");
    if (!fp)
    {
        return 0;
    }
    fclose(fp);
    return 1;
}
#endif    // __UTILS_HPP__