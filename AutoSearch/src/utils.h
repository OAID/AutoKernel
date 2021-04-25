#include <iostream>
#include <functional>
#include <string>
#include <vector>
#include <stdlib.h>

inline int GetArg(const std::vector<int> &args, size_t index, int default_value = 0) {
    return index < args.size() ? args[index] : default_value;
}

inline std::vector<int> GetArgsFromEnv() {
    std::vector<int> ret;
    if (const char* env_p = std::getenv("HL_APP_ARGS")) {
        std::string val(env_p);
        size_t offset = 0;
        auto pos = val.find(',', offset);
        while (pos != std::string::npos) {
            ret.push_back(std::stoi(val.substr(offset, pos - offset)));
            offset = pos + 1;
            pos = val.find(',', offset);
        }
        ret.push_back(std::stoi(val.substr(offset, val.size() - offset)));
    } else {
        std::cerr << "Cannot load arguments from environment variable HL_APP_ARGS" << std::endl;
        exit(-1);
    }
    return ret;
}

inline double benchmark();
