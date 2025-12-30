import client from "./client";

export type Topic = {
  id: number;
  name: string;
  description?: string;
  postCount: number;
};

export type Page<T> = {
  content: T[];
  totalElements: number;
  totalPages: number;
  number: number;
  size: number;
};

export type Post = {
  id: number;
  topicId: number;
  topicName: string;
  title: string;
  content: string;
  authorUsername: string;
  createdAt: string;
  updatedAt: string;
  commentCount: number;
};

export type Comment = {
  id: number;
  postId: number;
  authorUsername: string;
  content: string;
  createdAt: string;
};

export const forumApi = {
  listTopics: async () => {
    const res = await client.get<Topic[]>("/api/forum/topics");
    return res.data;
  },

  listPostsByTopic: async (topicId: number, page = 0, size = 10, q?: string) => {
    const res = await client.get<Page<Post>>(`/api/forum/topics/${topicId}/posts`, {
      params: { page, size, q: q?.trim() || undefined },
    });
    return res.data;
  },


  createPost: async (payload: { topicId: number; title: string; content: string }) => {
    const res = await client.post<Post>("/api/forum/posts", payload);
    return res.data;
  },

  getPost: async (postId: number) => {
    const res = await client.get<Post>(`/api/forum/posts/${postId}`);
    return res.data;
  },

  listComments: async (postId: number, page = 0, size = 30) => {
    const res = await client.get<Page<Comment>>(
      `/api/forum/posts/${postId}/comments`,
      { params: { page, size } }
    );
    return res.data;
  },

  createComment: async (postId: number, payload: { content: string }) => {
    const res = await client.post<Comment>(`/api/forum/posts/${postId}/comments`, payload);
    return res.data;
  },
  getMyRecentPosts: async (page = 0, size = 5) => {
    const res = await client.get<Page<Post>>("/api/forum/posts/mine", {
      params: { page, size },
    });
    return res.data;
  },

  createTopic: async (payload: { name: string; description?: string }) => {
    const res = await client.post<Topic>("/api/forum/topics", payload);
    return res.data;
  },


};
