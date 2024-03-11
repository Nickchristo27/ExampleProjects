//Nicholas Christophides 113319835

import java.util.ArrayList;

public class MusicService {
	private static ArrayList<User> userList = new ArrayList<User>();
	
	public static void addUser(User user_) { userList.add(user_); }
	
	public void removeUser(User user_) { userList.remove(user_); }
	
	public ArrayList<Playlist> getPlaylists(User user_){ return user_.getPlaylists(); }
}
