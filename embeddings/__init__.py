# Embeddings 模块
# 向量化工具
#
# 三个向量化服务，各司其职：
# - TextEmbedding      → 文本 → 百炼 text-embedding-v4 → 1024维
# - ImageEmbedding     → 图片 → 百炼 multimodal-embedding-v1 → 1024维
# - MultimodalEmbedding → 图文混合 → 组合上述两者，统一入口
#
# 使用方式：
#   from embeddings.text_embedding import get_text_embedding
#   from embeddings.image_embedding import get_image_embedding
#   from embeddings.multimodal_embedding import get_multimodal_embedding
