add_executable(whisper-udp-stream udp-stream.cpp)

target_link_libraries(whisper-udp-stream PRIVATE
    whisper
    common
)

if (UNIX)
    target_link_libraries(whisper-udp-stream PRIVATE
        pthread
    )
endif()
