namespace detail {
struct CompositeTypeStorage : public TypeStorage {
  struct KeyTy {
    ArrayRef<Type> members;
    StringAttr name;
    bool incomplete;

    KeyTy(ArrayRef<Type> members, StringAttr name, bool incomplete)
        : members(members), name(name), incomplete(incomplete) {}
  };

  ArrayRef<Type> members;
  StringAttr name;
  bool incomplete;

  CompositeTypeStorage(ArrayRef<Type> members, StringAttr name, bool incomplete)
      : members(members), name(name), incomplete(incomplete) {}

  KeyTy getAsKey() const { return KeyTy(members, name, incomplete); }

  bool operator==(const KeyTy &key) const {
    if (name)
      return name == key.name;
    return (members == key.members) && (name == key.name) &&
           (incomplete == key.incomplete);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    if (key.name)
      return llvm::hash_combine(key.name);
    return llvm::hash_combine(key.members, key.incomplete);
  }

  static CompositeTypeStorage *construct(TypeStorageAllocator &allocator,
                                         const KeyTy &key) {
    return new (allocator.allocate<CompositeTypeStorage>())
        CompositeTypeStorage(allocator.copyInto(key.members), key.name,
                             key.incomplete);
  }

  /// Mutates the members and attributes an identified struct.
  ///
  /// Once a record is mutated, it is marked as complete, preventing further
  /// mutations. Anonymous structs are always complete and cannot be mutated.
  /// This method does not fail if a mutation of a complete struct does not
  /// change the struct.
  LogicalResult mutate(TypeStorageAllocator &allocator,
                       ArrayRef<Type> members) {
    // Anonymous structs cannot mutate.
    if (!name)
      return failure();

    // Mutation of complete structs are allowed if they change nothing.
    if (!incomplete)
      return mlir::success(this->members == members);

    // Mutate incomplete struct.
    this->members = allocator.copyInto(members);

    incomplete = false;
    return success();
  }
};
} // namespace detail
