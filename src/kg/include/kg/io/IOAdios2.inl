
namespace kg
{
namespace io
{

inline IOAdios2::IOAdios2()
{
  try {
    ad_.emplace("adios2cfg.xml", MPI_COMM_WORLD);
  } catch (const std::exception&) {
    ad_.emplace(MPI_COMM_WORLD);
  }
}

inline IOAdios2::IOAdios2(const std::string& config)
{
  try {
    ad_.emplace(config, MPI_COMM_WORLD);
  } catch (const std::exception&) {
    ad_.emplace(MPI_COMM_WORLD);
  }
}

inline File IOAdios2::openFile(const std::string& name, const Mode mode,
                               MPI_Comm comm, const std::string& io_name)
{
  return File{new FileAdios2{*ad_, name, mode, io_name}};
}

inline Engine IOAdios2::open(const std::string& name, const Mode mode,
                             MPI_Comm comm, const std::string& io_name)
{
  return {openFile(name, mode, comm, io_name), comm};
}

} // namespace io
} // namespace kg
